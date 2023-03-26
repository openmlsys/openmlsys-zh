## 总结

-   大型机器学习模型的出现带来了对于算力和内存需求的快速增长，催生了分布式训练系统的出现。

-   分布式训练系统的设计往往遵循"分而治之"的设计思路。

-   利用分布式训练系统，人们可以显著提升性能，经济性，并且帮助抵御硬件故障。

-   分布式训练系统可以通过数据并行增加设备来提升算力。

-   当单节点内存不足时，我们可以通过模型并行来解决单设备内存不足。模型并行有两种实现方式：算子内并行和算子间并行。

-   大型模型并行系统容易出现设备使用空洞，而这种空洞可以通过流水线并行解决。

-   分布式训练系统往往运行在商用数据中心之中，数据中心网络无法提供充足的网络带宽来传输大量训练中生成的梯度。

-   为了提供海量的带宽，机器学习集群拥有异构的网络：以太网，机内网络（NVLink）和InfiniBand。

-   为了解决单节点瓶颈，我们可以使用Allreduce来分摊梯度聚合过程中的计算和通讯开销。

-   参数服务器可以帮助机器学习集群实现计算-存储的分离，从而更好的支持大型稀疏模型。

-   参数服务器常用数据副本技术解决数据热点问题，同时它们也可以被用来解决同步训练系统中常见的掉队者问题。


## 扩展阅读

- 分布式机器学习系统：[综述](https://dl.acm.org/doi/abs/10.1145/3377454)

- 利用集合通信支持并行训练的实践：[Horovod](https://arxiv.org/abs/1802.05799)

- AllReduce的工程实现细节：[树形结构](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)，[环形结构](https://github.com/baidu-research/baidu-allreduce)，[二维环面结构](https://arxiv.org/abs/1811.05233)，以及[CollNet算法](https://github.com/NVIDIA/nccl/issues/320)

- 流水线并行的实践：[gPipe](https://arxiv.org/abs/1811.06965)

- 在大规模数据并行下的实践：[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

- 模型并行在超大模型上的实践：[ZeRO](https://arxiv.org/abs/1910.02054)

- 最后，在讨论集合通信时，经常可以看到一些关于底层通信接口的专业术语，例如以太网，Infiniband 等。这里给出一些常见术语的具体定义：

  * [以太网（Ethernet)](https://web.archive.org/web/20181222184046/http://www.mef.net/Assets/White_Papers/Metro-Ethernet-Services.pdf)
  * [NVLink](https://devblogs.nvidia.com/parallelforall/how-nvlink-will-enable-faster-easier-multi-gpu-computing/)
  * [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/cn/hpc/efa/)
  * [Infiniband](https://www.infinibandta.org/about-infiniband/)
  * [RDMA](http://reports.ias.ac.in/report/12829/understanding-the-concepts-and-mechanisms-of-rdma)
  * [RoCE](https://www.roceinitiative.org/about-overview/)
  * [IPoIB](https://www.ibm.com/docs/en/aix/7.2?topic=protocol-internet-over-infiniband-ipoib)

## 参考文献

:bibliography:`../references/distributed.bib`