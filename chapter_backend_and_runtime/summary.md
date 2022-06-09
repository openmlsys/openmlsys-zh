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
-   本章提供了一节扩展部分内容，简要地介绍了当前业界面临的算子编译器部分的几大挑战，包括不限于：调度策略、切分参数、多芯片适配和增强表达能力。目前这些问题尚未收敛。本小节将它们逐一介绍，以期对相关问题的解决和收敛起到抛砖引玉的作用。

## 扩展阅读

-   内存分配作为机器学习后端的重要部分，建议阅读 [Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)、 [Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616)。
-   对于运行时的调度以及执行，建议阅读 [A Lightweight Parallel and Heterogeneous Task Graph Computing System](https://arxiv.org/abs/2004.10908)、 [Dynamic Control Flow in Large-Scale Machine Learning](https://arxiv.org/abs/1805.01772)、[DEEP LEARNING WITH DYNAMIC COMPUTATION GRAPHS](https://arxiv.org/abs/1702.02181)。
-   对于扩展部分的算子编译器，建议阅读词条：[领域特定语言（Domain Specific Language, DSL）](https://en.wikipedia.org/wiki/Domain-specific_language)、[局部性概念](https://en.wikipedia.org/wiki/Locality_of_reference)、[并行性概念](https://en.wikipedia.org/wiki/Parallel_computing)、[调度原语（schedule primitives）](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html?highlight=schedule%20primitives)、[单指令多数据（Single instruction, multiple data, SIMD）](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)、[单指令多线程（Single instruction, multiple threads, SIMT）](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)、[CUDA](https://en.wikipedia.org/wiki/CUDA)、[NEON](https://en.wikipedia.org/wiki/ARM_architecture_family#Advanced_SIMD_(Neon))、[图层编译器Relay的表达能力](https://tvm.apache.org/docs/reference/langref/relay_expr.html)、[混合脚本(hybrid script)](https://tvm.apache.org/docs/reference/langref/hybrid_script.html)；论文：自动调度的算法Ansor :cite:`zheng2020ansor`、多面体模型编译（polyhedron compilation）技术 :cite:`grosser2011polly`。另外，本小节中出现的几个重要开源项目为：[TVM](https://github.com/apache/tvm)、[MLIR](https://mlir.llvm.org/)、[LLVM](https://github.com/llvm/llvm-project)、[pytorch](https://github.com/pytorch/pytorch)、[torch-mlir](https://github.com/llvm/torch-mlir)。