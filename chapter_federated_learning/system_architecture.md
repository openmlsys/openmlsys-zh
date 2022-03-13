## 系统架构

联邦学习系统由调度器模块、服务器模块和客户端模块三个部分组成，其系统架构如 :numref:`ch10-federated-learning-architecture`所示。其中：

- 联邦学习调度器：

  联邦学习调度器（FL-Scheduler）协助集群组网，并负责管理面任务的下发。

-  联邦学习服务器：

  联邦学习服务器（FL-Server）提供客户端选择、限时通信、分布式联邦聚合功能。FL-Server需要具备支持端云千万台设备的能力以及边缘服务器的接入和安全处理的逻辑。

- 联邦学习客户端：

  联邦学习客户端（FL-Client）负责本地数据训练，并在和FL-Server进行通信时，对上传权重进行安全加密。

![联邦学习系统架构图](../img/ch10/ch10-federated-learning-architecture.png)

:label:`ch10-federated-learning-architecture`

