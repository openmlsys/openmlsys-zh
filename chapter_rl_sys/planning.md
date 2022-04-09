规划系统
========

规划不仅包含运动路径规划，还包含高级任务规划 [@9712373]。其中，运动规划是机器人技术的核心问题之一，应用范围从导航到复杂环境中的操作。它具有悠久的研究历史，方法需要有概率完整性和最优性的保证。然而，当经典运动规划在处理现实世界的机器人问题（在高维空间中）时，挑战仍然存在。研究人员在继续开发新算法来克服与这些方法相关的限制，包括优化计算和内存负载、更好的规划表示和处理维度灾难等。

相比之下，机器学习的最新进展为机器人专家研究运动规划问题开辟了新视角：经典运动规划器的瓶颈可以以数据驱动的方式解决；基于深度学习的规划器可以避免几何输入的局限性，例如使用视觉或语义输入进行规划等。最近的工作有：基于深度神经网络的四足机器人快速运动规划框架 [@jangdeep]，通过贝叶斯学习进行运动规划 [@quintero2021motion]，通过运动规划器指导的视觉运动策略学习 [@kadubandimotion]。ML4KP [@ML4KP]是一个用于有效运动动力学运动规划的C++库，该库可以轻松地将机器学习方法集成到规划过程中。
自动驾驶领域和行人和车辆轨迹预测 [@qiu2021egocentric]方面也涌现出使用机器学习解决运动规划的工作，比如斯坦福大学提出Trajectron++ [@salzmann2020trajectron++]。强化学习在规划系统上也有重要应用 [@aradi2020survey; @sun2021adversarial],比如基于MetaDrive模拟器 [@li2021metadrive]，最近有一些关于多智能体强化学习，多智能体车流模拟、Social
Behavior分析 [@peng2021learning]，Safe
RL [@peng2021safe]的工作，以及拓展到由真人专家在旁边监督，出现危险的时候接管的Safe
RL工作（Online Imitation Learning、Offline
RL） [@li2021efficient]，样本效率极高，是单纯强化学习算法的50倍。为了更好地说明强化学习是如何应用在自动驾驶中的，图 [\[fig:rl\_ad\]](#fig:rl_ad){reference-type="ref"
reference="fig:rl_ad"}展示了一个基于深度强化学习的自动驾驶POMDP模型。

![**基于深度强化学习的自动驾驶POMDP模型** [@aradi2020survey]](../img/ch13/rl_ad){width="\linewidth"}

[\[fig:rl\_ad\]]{#fig:rl_ad label="fig:rl_ad"}
