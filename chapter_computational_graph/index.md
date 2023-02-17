# 计算图

上一章节展示了如何高效编写机器学习程序，那么下一个问题就是：机器学习系统如何高效地在硬件上执行这些程序呢？这一核心问题又能被进一步拆解为：如何对机器学习程序描述的模型调度执行？如何使得模型调度执行更加高效？如何自动计算更新模型所需的梯度？解决这些问题的关键是计算图（Computational Graph）技术。为了讲解这一技术，本章将详细讨论计算图的基本组成、自动生成和高效执行中所涉及的方法。

本章的学习目标包括：
-   掌握计算图的基本构成。
-   掌握计算图静态生成和动态生成方法。
-   掌握计算图的常用执行方法。

```toc
:maxdepth: 2

background_and_functionality
components_of_computational_graph
generation_of_computational_graph
schedule_of_computational_graph
summary
```