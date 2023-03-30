## 总结

-   编译器后端主要负责计算图优化、算子选择、内存分配这三个任务。

-   计算图优化是在不影响模型的数值特性的基础上，通过图变换达到减少资源开销、适配硬件的执行能力、提升执行性能的目的。

-   计算图优化主要分为硬件通用优化和特定硬件优化，例如与硬件无关的算子内存IO优化和为了适配特定硬件指令限制而做的子图变换。

-   算子选择是为IR图中的每个计算节点选择一个最适合在设备上执行的算子。

-   数据存在多种存储格式和计算精度，不同的存储格式和计算精度在不同场景下对算子计算性能有较大的影响，所以算子选择需要综合考虑各方面影响选择最优的算子。

-   经过计算图优化和算子选择之后，得到了最终的IR。基于最终的IR，需要为算子的输入输出Tensor分配内存，然后加载算子到硬件上执行。

-   内存复用是一个重要的内存分配优化手段，可以让设备上容纳更大的网络模型。

-   将通信算子的内存进行融合，可以提高通信的效率；合理分配In-Place算子的内存，可以节省内存使用并且提高计算效率。

-   运行时对于算子的执行可以分为单算子调度和计算图调度两种模式，而在计算图调度模式中，根据具体硬件的能力又可以分为交互式执行和下沉式执行两种方式，交互式执行具备更多的灵活性，下沉执行可以获得更好的计算性能。

- 算子编译器是优化硬件性能的关键组件。其中，调度策略的优化和基于多面体模型算法的优化是两个关键技术。

## 扩展阅读

-   内存分配作为机器学习后端的重要部分，建议阅读 [Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)、 [Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616)。
-   对于运行时的调度以及执行，建议阅读 [A Lightweight Parallel and Heterogeneous Task Graph Computing System](https://arxiv.org/abs/2004.10908)、 [Dynamic Control Flow in Large-Scale Machine Learning](https://arxiv.org/abs/1805.01772)、[DEEP LEARNING WITH DYNAMIC COMPUTATION GRAPHS](https://arxiv.org/abs/1702.02181)。
- 算子编译器是本书的扩展部分，建议阅读提出计算与调度分离的论文：[Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](https://dl.acm.org/doi/abs/10.1145/2499370.2462176)，以及介绍调度空间优化的论文[Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762)和[Polly - Polyhedral optimization in LLVM](https://arxiv.org/abs/2105.04555)。