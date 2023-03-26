## 总结

- 现代机器学习系统需要兼有易用性和高性能，因此其一般选择Python作为前端编程语言，而使用C和C++作为后端编程语言。

- 一个机器学习框架需要对一个完整的机器学习应用工作流进行编程支持。这些编程支持一般通过提供高层次Python API来实现。

- 数据处理编程接口允许用户下载，导入和预处理数据集。

- 模型定义编程接口允许用户定义和导入机器学习模型。

- 损失函数接口允许用户定义损失函数来评估当前模型性能。同时，优化器接口允许用户定义和导入优化算法来基于损失函数计算梯度。

- 机器学习框架同时兼有高层次Python API来对训练过程，模型测试和调试进行支持。

- 复杂的深度神经网络可以通过叠加神经网络层来完成。

- 用户可以通过Python API定义神经网络层，并指定神经网络层之间的拓扑来定义深度神经网络。

- Python和C之间的互操作性一般通过CType等技术实现。

- 机器学习框架一般具有多种C和C++接口允许用户定义和注册C++实现的算子。这些算子使得用户可以开发高性能模型，数据处理函数，优化器等一系列框架拓展。

## 扩展阅读

- MindSpore编程指南：[MindSpore](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/index.html)
- Python和C/C++混合编程：[Pybind11](https://pybind11.readthedocs.io/en/latest/basics.html#creating-bindings-for-a-simple-function)