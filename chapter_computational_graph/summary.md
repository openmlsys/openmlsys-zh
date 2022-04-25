## 总结

-   为了兼顾编程的灵活性和计算的高效性，设计了基于计算图的深度学习框架。

-   计算图的基本数据结构是张量，基本运算单元是算子。

-   计算图可以表示机器学习模型的计算逻辑和状态，利用计算图分析图结构并进行优化。

-   计算图是一个有向无环图，图中算子间可以存在直接依赖和间接依赖关系，或者相互关系独立，但不可以出现循环依赖关系。

-   可以利用控制流来改变数据在计算图中的流向，常用的控制流包括条件控制和循环控制。

-   计算图的生成可以分为静态生成和动态生成两种方式。

-   静态图计算效率高，内存使用效率高，但调试性能较差，可以直接用于模型部署。

-   动态图提供灵活的可编程性和可调试性，可实时得到计算结果，在模型调优与算法改进迭代方面具有优势。

-   利用计算图和算子间依赖关系可以进行模型中的算子执行调度问题。

-   根据计算图可以找到相互独立的算子进行并发调度，提高计算的并行性。而存在依赖关系的算子则必须依次调度执行。

-   计算图的训练任务可以使用同步或者异步机制，异步能够有效提高硬件使用率，缩短训练时间。

## 扩展阅读

-   计算图是计算框架的核心理念之一，了解主流计算框架的设计思想，有助于深入掌握这一概念，建议阅读 [TensorFlow 设计白皮书](https://arxiv.org/abs/1603.04467)、 [PyTorch计算框架设计论文](https://arxiv.org/abs/1912.01703)、[MindSpore技术白皮书](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/white_paper/MindSpore_white_paperV1.1.pdf)。
-   图外控制流直接使用前端语言控制流，熟悉编程语言即可掌握这一方法，而图内控制流则相对较为复杂，建议阅读[TensorFlow控制流](http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf)论文。
-   动态图和静态图设计理念与实践，建议阅读[TensorFlow Eager 论文](https://arxiv.org/pdf/1903.01855.pdf)、[TensorFlow Eager Execution](https://tensorflow.google.cn/guide/eager?hl=zh-cn)示例、[TensorFlow Graph](https://tensorflow.google.cn/guide/intro_to_graphs?hl=zh-cn)理念与实践、[MindSpore动静态图](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/design/dynamic_graph_and_static_graph.html)概念。