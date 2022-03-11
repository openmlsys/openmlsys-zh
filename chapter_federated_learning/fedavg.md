## 联邦平均算法

和传统分布式学习相比，联邦学习存在训练结点不稳定和通信代价大的难点。这些难点导致了联邦学习无法和传统分布式学习一样：在每次单步训练之后，同步不同训练结点上的权重。为了提高计算通信比并降低频繁通信带来的高能耗，谷歌公司提出了联邦平均算法（Federated Averaging，FedAvg）。图:numfef:`ch10-federated-learning-fedavg`展示了FedAvg的整体流程。在每轮联邦训练过程中，端侧进行多次单步训练。然后云侧聚合多个端侧权重，并取加权平均。

![联邦平均算法](../img/ch10/ch10-federated-learning-fedavg.png)

:label:`ch10-federated-learning-fedavg`

