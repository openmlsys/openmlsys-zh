## 算子编译器（扩展部分）

算子编译器，顾名思义，即对“单个算子”进行编译优化的工具。这里所谓的“单个算子”可以来自于整个神经网络中的一部分，也可以来自于通过[领域特定语言（Domain Specific Language, DSL）](https://en.wikipedia.org/wiki/Domain-specific_language)实现的代码。而所谓编译，通俗来说起到的是针对目标语言进行**表达**和**转换**。

从目的上来说，算子编译器致力于提高单个算子的**执行性能**。从工程实现上来说，算子编译器的输入一般为python等**动态语言**描述的张量计算，而输出一般为**特定AI芯片**上的可执行文件。

针对上述两个方面，我们继续进行深入思考。

### 如何提升算子的执行性能？

编译器如果不考虑优化和实际中芯片的体系结构特点，我们只需要按照算子表达式的**计算逻辑**，把输入进来的张量全部加载进计算核心里完成计算，之后再把计算结果从计算核心里面取出并保存下来即可。这里的**计算逻辑**指的就是基本数学运算（如加、减、乘、除）以及其他函数表达式（如卷积、转置、损失函数）等。

但事实上，我们有“处理器在短时间内重复访问同一内存位置时效率高”这一[局部性概念](https://en.wikipedia.org/wiki/Locality_of_reference)。基于局部性概念，我们希望尽量把需要重复处理的数据放在固定的内存位置，且这一内存位置离处理器越近越好，以通过提升访存速度而进行性能提升。另外，我们有“计算任务总量一定时，同时并行计算的任务量越多，总耗时最少”这一[并行性概念](https://en.wikipedia.org/wiki/Parallel_computing)。基于并行性概念，我们希望尽量使得输入数据能够切分成多个互相没有依赖关系的数据段，以通过并行地对它们进行计算而进行性能提升。且我们知道现代计算机中有[缓存（cache）](https://en.wikipedia.org/wiki/Cache_(computing))可以进一步提高访存速度，但是由于其造价高昂而往往容量很小。为了充分利用缓存，我们希望尽量把重要的、需要经常访存的数据存入缓存中，以通过高速访存而进行性能提升。以上种种在程序实际运行的时候针对数据做出的特殊操作，我们统称为**调度（schedule）策略**。

MIT CASIL组的[Jonathan Ragan-Kelley](http://people.csail.mit.edu/jrk/)在2013年发表的文章 :cite: `ragan2013halide`中给出了schedule的精确定义：

1. When and where should be the value at each coordinate in each function be computed?
2. Where should they be stored?
3. How long are values cached and communicated across multiple consumers, and when are they independently recomputed by each?

通俗理解，调度策略指的是：在编译阶段根据目标硬件体系结构的特点而设计出的一整套通过提升局部性和并行性而使得编译出的可执行文件在运行时性能最优的算法。这些算法并不会影响计算结果，只是干预计算过程，以达到提升运算速度的效果。

#### 针对调度策略的优化

在[TVM](https://github.com/apache/tvm)中，这一思想得到了发扬。这里我们以CPU为例，结合TVM，简要介绍其中几种基本调度策略组成的优化算法。

假设我们有以下形式为乘累加计算的输入代码。该计算逻辑为：张量A与张量B相乘后，结果累加到张量C中。

```c
for (m: int32, 0, 1024) {
  for (n: int32, 0, 1024) {
    C[((m*1024) + n)] = 0f32
      for (k: int32, 0, 1024) {
        let cse_var_2: int32 = (m*1024)
          let cse_var_1: int32 = (cse_var_2 + n)
            C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + k)]*B[((k*1024) + n)]))
      }
  }
}
```

假定数据类型为浮点型（float），此时张量A、B、C的大小均为`1024 * 1024`，三者占用的空间共为`1024 * 1024 * 3 * sizeof(float) = 12MB`。这远远超出了常见缓存的大小（如L1 cache为32KB）。因此按照此代码形式，要将整块张量A、B、C一起计算，只能放入离计算核更远的内存进行计算。其访存效率远低于缓存。

为了提升性能，我们提出使用平铺（tile），循环移序（reorder）和切分（split）的调度策略，对于`i`循环和`j`循环都按照大小为32的因子（factor）进行平铺，对于`k`循环按照大小为4的因子进行切分，最后将切出的`k`循环移动到合适位置。经过该策略优化后的代码如下：

```c
for (m.outer: int32, 0, 32) {
  for (n.outer: int32, 0, 32) {
    for (m.inner.init: int32, 0, 32) {
      for (n.inner.init: int32, 0, 32) {
        C[((((m.outer*32768) + (m.inner.init*1024)) + (n.outer*32)) + n.inner.init)] = 0f32
      }
    }
    for (k.outer: int32, 0, 256) {
      for (k.inner: int32, 0, 4) {
        for (m.inner: int32, 0, 32) {
          for (n.inner: int32, 0, 32) {
            let cse_var_3: int32 = (n.outer*32)
              let cse_var_2: int32 = ((m.outer*32768) + (m.inner*1024))
                let cse_var_1: int32 = ((cse_var_2 + cse_var_3) + n.inner)
                  C[cse_var_1] = (C[cse_var_1] + (A[((cse_var_2 + (k.outer*4)) + k.inner)]*B[((((k.outer*4096) + (k.inner*1024)) + cse_var_3) + n.inner)]))
          }
        }
      }
    }
  }
}
```

这里我们重点看第8-19行的核心计算语句。每次计算时只需要关注`m.inner * n.inner`构成的小块（block）即可，而其他的外层循环不会影响最内层小块的访存。其占用内存大小为`32 * 32 * 3 * sizeof(float) = 12KB`，足够放入cache中。

本示例参照TVM提供的[在CPU上优化矩阵乘运算的实例教程](https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html#)中的第一项优化，读者可深入阅读后续优化内容。

#### 多面体模型优化

然而业界另一编译器框架[MLIR](https://mlir.llvm.org/)做法则不同。它并没有明确提出调度抽象的概念，而是多面体模型编译（polyhedron compilation）技术 :cite:`grosser2011polly`主要对`for`循环进行优化。该算法的主要思想是针对输入代码的访存特点进行建模，调整 for 循环语句中的每一个实例的执行顺序（即调度方式），使得新调度下的 for 循环代码有更好的局部性和并行性。

假设我们有以下形式的代码：

```c
for (int i = 0; i < N; i++)
  for (int j = 1; j < N; j++)
    a[i+1][j] = a[i][j+1] - a[i][j] + a[i][j-1];
```

通过多面体模型算法先对此代码进行访存建模，再进行复杂的依赖分析和调度变换之后得到一个符合内存模型的最优解，生成的代码为：

```c
for (int i_new = 0; i_new < N; i_new++)
  for (int j_new = i+1; j_new < i+N; j_new++)
    a[i_new+1][j_new-i_new] = a[i_new][j_new-i_new+1] - a[i_new][j_new-i_new] + a[i_new][j_new-i_new-1];
```

观察得到的代码，发现从直觉上优化后的代码较为复杂。但是仅凭肉眼很难发现其性能优势之处。仍需对此优化后的代码进行如算法描述那样建模，并分析依赖关系后得出结论：经过算法优化后解除了原代码中的循环间的依赖关系，从而提高了并行计算的机会。该算法较为复杂，限于篇幅，在这里不再详细展开。读者可移步到笔者专门为此例写的文章-[深度学习编译之多面体模型编译——以优化简单的两层循环代码为例](https://zhuanlan.zhihu.com/p/376285976)详读。

除了核心的优化任务之外，还有两方面值得简要一提。

### 如何适配不同的AI芯片？

一般意义上来说，通用编译器的设计会尽量适配多种后端。如此一来，在面临不同体系结构特点和不同编程模型的多种后端时，算子编译器承受了相当大的压力。

当下的AI芯片中，常见的编程模型分为：[单指令多数据（Single instruction, multiple data, SIMD）](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)，即单条指令一次性处理大量数据；[单指令多线程（Single instruction, multiple threads, SIMT）](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)，即单条指令一次性处理多个线程的数据。前者对应的是带有向量计算指令的芯片，比如华为的昇腾系列芯片等；后者对应的是带有明显的线程分级的芯片，比如英伟达的V系列和A系列芯片等。另外，也有一些芯片开始结合这两种编程模型的特点，像寒武纪的思元系列芯片，既有类似线程并行计算的概念，又有向量指令的支持。针对不同的编程模型，算子编译器在进行优化（如向量化等）时的策略也会有所不同。

### 如何表达的准确而完整？

假设我们已经解决了上述调度优化和多芯片适配的问题，看起来似乎一切已经非常完美。然而我们还需要考虑到在实际场景中，AI编译器的输入常以[pytorch](https://github.com/pytorch/pytorch)代码居多，即此输入带有大量python的灵活表达方式（包括而不限于索引、view语义等）。另外在检测网络中，输入算子往往还有大量的控制流语句。此外，还经常可以看到神经网络中存在许多的动态形状问题，即网络中的算子形状会受网络迭代次数和控制流等条件的影响。这些都对算子编译器前端的表达能力提出了很高的要求。

在实际工程实践中，我们发现大量的长尾分布般不常见但性能很差的算子（后文简称为长尾算子）往往是整体网络训练或推理的瓶颈点。而这些长尾算子大都是由于其出现频次低而不至于实现在计算库中。同时其语法过于灵活或存在大量的控制流语句以及动态形状问题而难以被目前的算子编译器前端充分表达出来，因此也难以通过算子编译器进行优化加速。于是，这些长尾算子只好以运行速度较慢的python解释器或者虚拟机的方式执行，从而成为整个网络中的性能瓶颈。此时，提高算子编译器前端的表达能力就成为了重中之重。
