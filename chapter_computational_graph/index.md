# 计算图

在上一章节中，我们展示了用户利用机器学习框架所编写的程序。这些用户程序包含了对于训练数据，模型和训练过程的定义。然而为了运行这些程序，机器学习系统依然需要解决诸多问题，包括：如何高效执行一个复杂的机器学习模型？如何识别出机器学习模型中需要训练的参数？如何自动计算更新模型所需的梯度？为了解决这些问题，现代机器学习框架实现了*计算图*(Computational
graph)这一技术。在本章中，我们详细讨论计算图的基本组成，生成和执行等关键设计。本章的学习目标包括：

-   掌握计算图的基本构成。

-   掌握计算图静态生成和动态生成两种方法。

-   掌握计算图的常用执行方法。

```toc
:maxdepth: 2

background_and_functionality
components_of_computational_graph
generation_of_computational_graph
schedule_of_computational_graph
summary
```