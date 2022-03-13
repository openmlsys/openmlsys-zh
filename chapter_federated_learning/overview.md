## 概述

为了解决“隐私保护”与“数据孤岛”这两大难题，联邦学习（Federated Learning，FL）应运而生。联邦学习的概念最早在2016年被提了出来，能有效帮助多个机构在满足用户隐私保护、数据安全和政府法规的要求下，进行数据使用和机器学习建模。在联邦学习场景中，每个用户，被定义为客户端这一角色，使用各自的本地数据进行训练、模型更新、权重上传，并在中央服务器的协调下，多个客户端协作建立机器学习模型。

根据数据分布的不同，联邦学习可以分为跨设备（cross-device）与跨组织（cross-silo）联邦学习。一般而言，跨组织联邦学习的用户一般是企业、机构单位级别的，而跨设备联邦学习针对的则是便携式电子设备、移动端设备等。 :numref:`ch10-federated-learning-different-connection`展示了两者的区别和联系：

![跨设备和跨组织联邦学习的区别和联系](../img/ch10/ch10-federated-learning-different-connection.png)

:label:`ch10-federated-learning-different-connection`

下面就联邦学习中的要点：系统架构、联邦平均算法、隐私加密算法以及实际部署时的挑战进行详细描述。