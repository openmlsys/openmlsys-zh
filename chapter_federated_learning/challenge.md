## 实际部署时的挑战

### 部署流程

随着端上算力的提升和算法的快速发展，跨设备联邦学习在智能手机或智能设备中的应用越来越广泛。其主要流程如图:numfef:`ch10-federated-learning-flow`所示，可分为如下几步：

1. FL-Client选择：FL-Client主动向FL-Server发起参与联邦学习的请求。FL-Server根据配置筛选出满足条件的FL-Client。选择成功后FL-Server下发模型以及相关联邦学习配置。
2. FL-Client训练：在终端设备上进行本地模型训练。
3. 联邦聚合：FL-Client上传模型权重到FL-Server，并由FL-Server选择聚合算法进行计算，并对某些数据进行持久化存储。
4. FL-Client模型更新：终端向FL-Server查询联邦聚合后的模型等数据。
5. 重复1-4步，完成联邦学习任务。

![跨设备联邦学习流程图](../img/ch10/ch10-federated-learning-flow.png)

:label:`ch10-federated-learning-flow`

### 部署挑战

然而，由于跨设备联邦学习的特殊性，其挑战主要包括：

1. 跨设备联邦学习的FL-Client的网络连接常常是不稳定的，在任何时候都只有部分FL-Client可用。

2. 跨设备联邦学习往往是大规模并行的场景，通信时间会成为瓶颈。当千万级FL-Client同时进行联邦学习请求时，需要大量网络带宽的支持，并且单台FL-Server必然承接不住庞大的数据和参数。

### 解决方案

为了解决跨设备联邦学习带来的挑战，MindSpore Federated Learning给出两个解决方案：

1. 限时通信：在FL-Server和FL-Client建立连接后，启动全局的计时器和计数器。当预先设定的时间窗口内的FL-Server接收到FL-Client训练后的模型参数满足初始接入的所有FL-Client的一定比例后，就可以进行聚合。若时间窗内没有达到比例阈值，则进入下一轮迭代。保证即使有海量FL-Client接入的情况下，也不会由于个别FL-Client训练时间过长或掉线导致的整个联邦学习过程卡死。
2. 松耦合组网：使用FL-Server集群。每个FL-Server接收和下发权重给部分FL-Client，减少单个FL-Server的带宽压力。此外，支持FL-Client以松散的方式接入。任意FL-Client的中途退出都不会影响全局任务，并且FL-Client在任意时刻访问任意FL-Server都能获得训练所需的全量数据。