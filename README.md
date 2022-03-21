# 机器学习系统：设计和实现

本开源项目试图给读者讲解现代机器学习系统的设计原理和实现经验。

[机器学习系统：设计和实现](https://openmlsys.github.io/) |  [PDF](https://pan.baidu.com/s/19IL6bt5Kh78QDJJzNfuuZA?pwd=cbjb)  

## 发布

- 17/03/2022: 本书处于勘误阶段。如发现文字和图片错误，可创建Issue并@[章节编辑](info/editors.md)。我们非常欢迎社区提交PR直接勘误。

## 适用读者

本书的常见读者包括：

-   **学生：**
    随着大量机器学习课程在大学中的普及，学生已经开始掌握大量机器学习的基础理论和神经网络的实现。然而，需要训练出可以实际应用的机器学习模型，需要对现代机器学习系统有充分的认识。

-   **科研人员：**
    研发新型的机器学习模型不仅仅需要会使用基础的机器学习系统接口。同时，新型的模型需要给系统提供新的自定义算子（Custom
    Operators），又或者是会利用高级的分布式执行算子来实现大模型的开发。这一系列需求都需要对底层系统具有充分认识。

-   **开发人员：**
    大量的数据和AI驱动的公司都部署了机器学习基础设施。这一设施的核心就是机器学习系统。因此了解机器学习系统有助于开发人员对于系统性能调优，以定位问题，并且根据业务需求对机器学习系统进行深度定制。

## 内容介绍

现代机器学习框架具有复杂的内部架构和繁多的外部相关组件。在本书中，我们将对其细致拆分，深入解读：

基础：

-   **编程接口：** 为了支持海量应用，机器学习框架的编程接口设计具有大量的设计哲学，在易用性和性能之间取得平衡。本书将讲述编程接口的演进，机器学习工作流，定义深度学习模型，以及用C/C++进行框架开发。

-   **计算图：** 机器学习框架需要支持自动微分，硬件加速器，多编程前端等。实现这些支持的核心技术是：计算图（Computational Graph）。本书将讲述计算图的基本构成，生成方法和调度策略。

性能进阶：

-   **编译器前端：**
    机器学习框架需要利用编译器前端技术对计算图进行功能拓展和性能优化。本书将讲述常见的前端技术，包括类型推导，中间表示（Intermediate Representation），自动微分等。

-   **编译器后端和运行时：**
    机器学习框架的一个核心目标是：如何充分利用异构硬件。这其中会涉及编译器后端技术，以及将计算图算子（Operator）调度到硬件上的运行时（Runtime）。本书将讲述计算图优化，算子选择，内存分配和计算调度与执行。

-   **硬件加速器：**
    机器学习框架的基本运行单元是算子，而算子的实现必须充分利用硬件加速器（GPU和Ascend）的特性。本书将会讲述硬件加速器的基本构成原理和常见的高性能编程接口。

-   **数据处理框架：**
    机器学习框架会集成高性能框架来进行数据预处理。本书将会讲述这一类数据处理框架在设计中需要达到的多个目标：易用性，高效性，保序性，分布式等。

-   **模型部署：**
    在模型完成训练后，用户需要将模型部署到终端设备（如云服务器，移动终端和无人车）。这其中涉及到的模型转换，模型压缩，模型推理和安全保护等知识也会在本书中讨论。

-   **分布式训练：**
    机器学习模型的训练需要消耗大量资源。越来越多的机器学习框架因此原生支持分布式训练。在本书中我们将会讨论常见的分布式训练方法（包括数据并行，模型并行和流水线并行），以及实现这些方法的系统架构（包括集合通讯和参数服务器）。

功能拓展：

-   **深度学习推荐系统：** 推荐系统是目前机器学习应用最成功的领域之一。本书将会概括推荐系统的运作原理，详细描述大规模工业场景下的推荐系统架构设计。

-   **联邦学习系统：** 随着数据保护法规和隐私保护的崛起，联邦学习正成为日益重要的研究领域。本书将会介绍联邦学习的常用方法以及相关系统实现。

-   **强化学习系统：** 强化学习是走向通用人工智能的关键技术。本书将会介绍目前常见的强化学习系统（包括单智能体和多智能体等）。

-   **可解释性AI系统：** 随着机器学习在安全攸关（Safety-critical）领域的应用，机器学习系统越来越需要对决策给出充分解释。本书将会讨论可解释AI系统的常用方法和落地实践经验。

我们在持续拓展拓展本书的内容，如元学习系统，自动并行，深度学习集群调度，绿色AI系统，图学习系统等。我们也非常欢迎社区对于新内容提出建议，贡献章节。

## 构建指南

请参考[构建指南](info/info.md)来了解如何构建本书的网页版本和PDF版本。

## 写作指南

我们欢迎大家来一起贡献和更新本书的内容。常见的贡献方式是提交PR来更新和添加Markdown文件。写作的风格和图片要求请参考[风格指南](info/style.md)。同时，机器学习领域涉及到大量的中英文翻译，相关的翻译要求请参考[术语指南](info/terminology.md)。
