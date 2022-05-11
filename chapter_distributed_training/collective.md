## 集合通讯

接下来，我们会讲解集合通讯 (Collective Communication) 在大型深度学习系统中的应用。作为并行计算中的一个重要概念，集合通信算子经常会被用来构建单程序流/多数据流编程环境（SPMD）中的许多交互模式。近年来，该领域无论是在对不同硬件架构的支持还是算法性能的发展上都成果颇丰，而因SPMD在大型深度学习系统中与数据并行的深厚联系，这些框架也在其中受益匪浅。因此，相比点对点 (Point-to-Point, p2p) 通信，我们有更大的兴趣去探讨如何高效地在数据中心（Data Centers）中实现这些集合通讯范式。首先，我们会介绍一些集合通讯中常见的算子，一个经典的利用All算法解决分布式训练系统中网络瓶颈的示例，探讨该算法在不同网络拓扑结构下的差异性以及一些重要指标（算法带宽，总线带宽）的计算方法，最后简略介绍现有机器学习系统对不同集合通讯算法的支持。

### 常见算子

在分布式内存模型（Distributed Memory Model）中，一些常见的进程间数据交互模式由硬件支持和并行算法的内在性质而涌现。因此，主流的并行计算架构标准（例如MPI）和机器学习系统的底层集合通讯库（例如gloo，NCCL）通常会支持数个经典的算子并针对其做优化，一般包括Broadcast，Reduce，AllGather，ReduceScatter 和 AllReduce。在一个基于 :cite:`Sanders2019-cq` 的简化理论模型下，可以对这些算子的特性进行简单的介绍并探讨具体的实现方法和计算开销。

#### 基本定义

首先，假定一个简化后的分布式内存模型：存在p个随机存取存储器（Random Access Machines, RAM）作为基础的处理单元（Processing Element, PE)，并由一个网络来连接所有的机器。每个处理单元有自己的独立内存，并且所有的处理单元间的通信都通过网络传输。同时，每个处理单元都知道自己的编号i，通常在1到p之间。
网络之间的通信在最底层的情况下均为点对点的全双工通信（full-duplex point-to-point communication)：

* 每次通信有且仅有一个发送者（sender）和一个接收者（receiver）。
* 在某个特定时刻，每个处理单元仅能至多发送或接收一个信息。但是，在网络中可以同时传输多个信息。每个处理单元也可以在发送一个信息的同时接受一个信息。
* 传输一个长度为l的信息会花费a+bl的时间，其中a代表延迟（latency），即单位信息通过网络从一个处理单元出发到达另一个处理单元所需的时间；b代表传输延迟（transmission delay），即把单位信息从处理单元中放到网络通信单元所需的时间。前者的大小一般取决于两个处理单元间的物理距离（同一个机架，同一个数据中心，横跨全球等），而后者的大小一般取决于通信网络的带宽。在这个模型下，假定所有处理单元之间的a和b均为恒定值。
* 通信可以指定一个发送者或者一个接收者：由于每个存储单元都由相对应的编号，我们可以定义两个函数send(i,l) 和receive(i,l)。其中send函数会把信息l从当前的处理单元发送至编号为i的处理单元，而receive函数会从编号为i的处理单元接受信息l。在调用send函数时，处理单元必须同时调用receive来保证编号为i的处理单元收到了该信息。因此，也可以说send和receive 同步（synchronize）了发送者和接受者。
* 作为拓展，我们也可以定义上述函数的一个变种：i = send(m) 和 i = receive(m)，即在传输信息时不规定发送着或接收者。这种情况下，网络中的任意一个处理单元都可以发送或接收该信息，而最终完成传输的处理单元的编号会作为函数的返回值。
* 虽然在现实生活中错误（fault）时常发生，但是在这个模型里，暂不考虑通信丢失（dropped message）和通信毁坏（corrupted message）的情况。

分布式内存模型中对于通信同步和传输的结合使得在这个理论模型下开发的代码更好维护。额外的，由于这个框架下提出的算法往往会产生一些很有规律的，包含了网络中所有处理单元的交互模式，通常会在最基础的点对点通信上维护一个算子库，用来归纳总结这些高校且更易于理解的算法，我们将其称为集合通信算子。

#### Broadcast
在SPMD中，最常见的一个交互模式经常是把一个位于处理单元i的信息发送到全部其他的节点，用于同步某种全局的变量或者参数。为此Broadcast算子可以定义为从编号为i的处理单元发送长度为l的信息给全部剩余的p-1个处理单元。在这里，一种简单的方法是在一个循环中使用p-1次send/receive来实现Broadcast，但这并不能很好的利用通信可并行化的特质（该算法只有(a+bl)(p-1)的线性时间复杂度）。为此，我们可以利用分治思想（divide-and-conquer）来对上述算法进行优化。假设所有的处理单元可以重新对编号进行排列，使得Broadcast的发送者为编号为1的处理单元。同时，为了简化计算过程，假设对于某个自然数n，p = 2^n。 现在，我们可以通过从1 向 p/2 发送一次信息来把问题转化为两个大小为p/2的子问题：编号为1的处理单元对1到p/2-1 的Broadcast，以及编号为p/2的处理单元对p/2到p的Broadcast。我们便可以通过在这两个子问题上进行递归来完成这个算法，并把临界条件定义为编号为i的处理单元在[i,i]这个区间里的Broadcast。此时，由于i本身已经拥有该信息，我们不需要做任何操作便可直接完成Broadcast。这个优化后的算法有(a+bl)log p 时间复杂度，因为在算法的每一阶段t，我们有2^t个计算单元在并行运行Broadcast算子。同时，算法一定会在log p 步之内结束。

#### Reduce
除了Broadcast，另一个常见的交互模式为程序试图概述在部分处理单元上得到的中间值。这时候，对于一个符合结合律（associative property）的算子f，我们可以定义Reduce算子，即将所有处理单元上的某个值两两配对重复应用该算子，并把最终结果出存在编号为i的计算单元上。常见的应用于Reduce中的算子有加和，乘积，最大值，最小值和平均值等。一个简易的Reduce的优化实现同样可以用分治思想来实现，即把1到p/2-1的Reduce结果存到编号为1的处理单元中，然后把p/2到p的Reduce结果存到p/2上。最后，我们可以把p/2的结果发送至1，执行f，并把最后的结果存至i。假设f的运行时间复杂度为常数并不改变其输出信息的长度l，Reduce的时间复杂度仍然为(a+bl)log p。

#### AllReduce
AllReduce算子为Reduce的一个变种，即将f的结果存至所有处理单元上。在这里，我们给出一个简化版的AllReduce 实现方式，即首先把最终值通过Reduce存到编号为1的处理单元，再将该值通过Broadcast广播到所有的处理单元上。在两个子算子都使用上述的算法情况下，AllReduce的时间复杂度仍为(a+bl)log p。

#### Gather
Gather算子尝试将每个处理单元上的信息全部聚合到编号为i的处理单元上，通常用于组装散落在每个处理单元上的独立信息。在聚合函数符合结合律的情况下，可以通过将其设为Reduce算子中的f来实现Gather算子。但是，在这种情况下，无论是基于链表还是数组的实现，在每一步的Reduce子问题中f的时间复杂度或输出长度l都发生了改变。因此，Gather并不具有先前Reduce或者Broadcast的时间复杂度，而是a log p + (p-1) bl。这是因为在算法的每一阶段t，我们传输的信息长度为l 2^t。

#### AllGather
相比起Gather，AllGather 算子会把聚合的结果存到所有的处理单元上。在这里，一个简单的做法是使用Gather和Broadcast把聚合结果先存到编号为1的处理单元中，再将其广播到剩余的处理单元上。这会产生一个a log p + (p-1) bl + (a+plb) log p的时间复杂度，因为在Broadcast时如果忽略链表/数组实现所带来的额外空间开销，每次通信的长度为pl而不是l。简化后，我们得到了一个a log p + plb log p 的时间复杂度。在一个基于超立方体的算法下，我们可以将其进一步优化到和Gather一样的a log p + (p-1) bl （:cite:`Sanders2019-cq`），然而由于篇幅问题便不再赘述。

#### Scatter
Scatter算子可以被视作Gather的逆运算：把一个存在于编号为i的处理单元上，长度为p（信息长度为pl）的链式数据结构L中的值分散到每个处理单元上，使得编号为i的处理单元会得到L[i]。我们可以通过模仿Gather算法来设计一个简易的Scatter实现：每一步的运算中，与其是聚集一半处理单元的结果，我们把现在的子链继续对半切分，并把前半段和后半段作为子问题进行递归。这时候，在算法的每一阶段t，我们传输的信息长度为l 2^(m-t)，其中m是算法总共运行的步骤，不会超过log p （见Broadcast）。最终，Scatter算子的检疫实现和Gather一样都有a log p + (p-1) bl 时间复杂度。在机器学习系统中，相比于链式数据结构，Scatter经常同时被用于可切分的数据结构，例如张量（tensor）在一个维度上的p等分等。

#### ReduceScatter
ReduceScatter算子可以视为Reduce 和 Scatter算子的组合体，即对于每个处理单元上分别拥有的一个链式/可切分数据结构，在通过f 概述后再重新分散到各个单元中。虽然我们已经知道了Reduce 和Scatter 各自的时间复杂度，但是在对ReduceScatter做时间复杂度分析是需要注意两部之间信息长度的变化：假设每个处理单元上的数据结构所需通信长度为pl，第一阶段的Reduce算法需要(a+plb)log p 时间复杂度。参照Scatter的分析，第二阶段的算子则需要 a log p + (p-1) bl 时间复杂度。综合下来，ReduceScatter 需要 a log p + plb log p 的时间复杂度，和AllGather相同。同时，运行ReduceScatter 和 AllGather的效果等同于运行一次AllReduce。

在SPMD中，通常还有一些额外的集合通信算子，如Prefix Sum，Barrier，All-to-all等，但由于篇幅限制以及与机器学习系统的有限联系，便不再赘述。最后，由于该模型下通信网络的拓扑结构较为简单，上文中呈现二叉树形的递归树也可以达到很好的实际运行速度。所有关于时间复杂度的分析也是基于这些相对简化的假设情况。后文中，我们将会用AllReduce举例介绍如何在更复杂的拓扑结构下设计不同的集合通信算子变种，并在时间复杂度之外去关注实际的通信量和运算时间。

### 在数据中心的梯度计算

接下来，我们将用一个示例来阐释集合通讯在机器学习系统中发挥的重要作用。

![数据中心](../img/ch09/ch10-datacentre.png)
:width:`800px`
:label:`ch10-datacentre`

 :numref:`ch10-datacentre` 描述了一个典型的用于深度学习模型训练的数据中心。数据中心中的训练服务器一般会有多个设备。如需增加服务器，我们会将多个训练服务器放置在一个机柜（Rack）上，同时接入一个架顶交换机（Top of Rack Switch）将其连接。在现有机柜满载的情况下，可以通过在架顶交换机间增加骨干交换机（Spine Switch）来接入新的机柜。通过这种方式，可以在数据中心内不断增加服务器，从而为神经网络的训练提供海量的算力和内存。目前的商用数据中心可拥有近百万台服务器。

在数据中心中训练大型神经网络的首要挑战是如何高效计算大量的平均梯度。假设给定一个千亿级别参数的神经网络（比如OpenAI 发布的大型语言模型GPT-3 :cite:`https://doi.org/10.48550/arxiv.2005.14165` 有将近1750亿参数），如果用32位浮点数来表达每一个参数，那么每一步训练中，一个数据并行模式下的模型副本（Model Replica）则需要生成700GB的本地梯度数据（即 175G $\times$ 4 bytes = 700GB）。假如有3个模型副本，那么至少需要传输1.4TB（即，700GB $\times$ $(3-1)$）的本地梯度数据（因为对于$N$个副本，只需传送其中的$N-1$个副本来完成计算）。当平均梯度计算完成后，需要进一步将其广播（Broadcast）到全部的模型副本（即1.4TB的数据）并更新其中的本地参数，从而确保模型副本不会偏离（Diverge）主模型中的参数。

当前的数据中心一般使用以太网（Ethernet）构建不同机柜之间的网络。主流的商用以太网链路带宽一般在10Gbps到25Gbps之间。利用以太网传输海量梯度会产生严重的传输延迟，从而降低模型训练的速度。新型深度学习训练集群（如英伟达的DGX系列机器）往往配置有更快的Inifiband。单个InfiniBand链路可以提供100Gbps或200Gbps的带宽。即使拥有这种高速网络，传输TB级别的本地梯度依然需要大量延迟（即使忽略网络延迟，1TB的数据在200Gbps的链路上传输也需要至少40秒）。

为了避免通过机间网络传输数据，现代深度学习服务器一般都会配备多个加速器（例如说，英伟达的DGX-3服务器会配备8个A100 GPU），而在一个服务器内的多个设备可以通过高速机内网络互联（如NVLink）。这种高速机内网络可以提供高达400GBps的带宽，从而让传输TB级别的数据成为可能。然而，受限于单个服务器的散热，成本和硬件等限制，通常无法在一个服务器内无限制的持续增加设备。因此，大型深度学习模型的训练仍需要多个服务器共同完成。在计算平均梯度时，服务器需要同时借助机间网络通信接口（以太网或InfiniBand）和机内通信接口（NVLink）。

### 基于AllReduce的梯度平均算法

我们将讨论如何利用AllReduce算子来实现数据中心中的高效梯度平均。首先，参照前文的分析，可以考虑一种简单的计算平均梯度的方法：在集群中分配一个设备来收集本地梯度，并在计算平均梯度后再将其广播到全部的设备。这种做法易于实现，但是引入了两个问题。首先，多台设备同时给该聚合设备发送数据时，聚合设备会因严重的带宽不足产生网络拥塞。其次，单台设备需要负担大量的梯度平均计算，而受限于单台设备上的有限算力，这种计算往往会受限于算力瓶颈。

![AllReduce初始状态和终止状态](../img/ch09/ch10-AllReduce-state.png)
:width:`800px`
:label:`ch10-AllReduce-state`

为了解决上述问题，可以引入AllReduce算子的Reduce-Broadcast实现来优化算法，其设计思路是：通过让全部的节点参与到梯度的网络通信和平均计算中，将巨大的网络和算力开销均摊给全部节点。这种做法可以解决先前单个梯度聚合节点的问题。假设有$M$个设备，每个设备存有一个模型副本，该模型由$N$个参数/梯度构成。那么按照AllReduce算子的要求，需要先将全部的参数按照设备数量切分成$M$个分区（Partition），使得每个分区具有$N/M$个参数。我们首先给出这个算法的初始和终止状态。如 :numref:`ch10-AllReduce-state` 所示，该例子含有3个设备。在每个设备有一个模型副本的情况下，这个副本有3个参数。那么按照AllReduce的分区方法，参数会被划分成3个分区（3个设备），而每一个分区则有1个参数（$N/M$，N代表3个参数，M代表3个设备）。在这个例子中，假定设备1拥有参数2,4,6，设备2拥有参数1,2,3，设备3拥有参数4,8,12，那么在使用AllReduce算子进行计算过后，全部的设备都将拥有梯度相加后的结果7,14,21，其中分区1的结果7是由3个设备中分区1的初始结果相加而成（7 = 1 + 2 + 4）。为了计算平均梯度，每个设备只需要在最后将梯度之和除以设备数量即可（分区1的最终结果为7除以3）。

![AllReduce算法的过程](../img/ch09/ch10-AllReduce-process.png)
:width:`800px`
:label:`ch10-AllReduce-process`

AllReduce算子会把梯度的计算拆分成$M-1$个Reduce算子和$M-1$个Broadcast算子（其中$M$是节点的数量）。其中，Reduce算子用于计算出梯度的和（Summation），Broadcast算子用于把梯度之和广播给全部的节点。为了说明这些算子的执行过程，可以参照 :numref:`ch10-AllReduce-process` 。AllReduce算子由Reduce算子开始，在第一个Reduce算子中，AllReduce算子会对全部节点进行配对（Pairing），让他们共同完成梯度相加的操作。在 :numref:`ch10-AllReduce-process` 的第一个Reduce算子中，设备1和设备2进行了配对共同对分区1的数据相加。其中，设备2把本地的梯度数据1发送给设备1，设备将接收到1和本地的分区1内的梯度数据：2进行相加，计算出中间（intermediate）梯度相加的结果：3。于此同时，设备1和设备3进行配对，共同完成对分区3的数据相加。而设备3和设备2进行配对，共同完成对于分区2的数据相加。

在上述Reduce的算子中，梯度的计算实现了以下几个特性:

-   **网络优化：**
    全部设备都同时在接收和发送数据，利用起了每个设备的入口（Ingress）和出口（Egress）带宽。因此AllReduce过程中可利用的带宽是$M \times B$，其中$M$是节点数量，$B$是节点带宽，从而让系统实现网络带宽上的可扩展性。

-   **算力优化：**
    全部设备的处理器都参与了梯度相加的计算。因此AllReduce过程中可利用的处理器是$M \times P$，其中$M$是节点数量，$P$是处理器数量，从而让系统实现计算上的可扩展性。

-   **负载均衡：**
    由于数据分区是平均划分的，因此每次设备分摊到的通讯和计算开销是相等的。

在接下来的Reduce算子中，AllReduce算法会对不同数据分区选择另外的配对方法。例如说，在 :numref:`ch10-AllReduce-process`  的第二个Reduce算子中，AllReduce算法会将：设备1和设备3进行配对，负责分区1的数据相加。将设备1和设备2进行配对，负责分区2。将设备2和设备3进行配对，负责分区3。在一个3个节点的AllReduce集群里，在2个Reduce算子完成后，我们就计算出了每个分区的数据相加结果（分区1的结果7此时在设备3上，分区2的结果14此时在设备1上，分区3的结果21此时在设备2上）。

接下来，AllReduce算法将进入Broadcast阶段。这一阶段的过程和Reduce算子类似，核心区别是节点进行配对后，他们不再进行数据相加，而是将Reduce的计算结果进行广播。在 :numref:`ch10-AllReduce-process`  中的第一个Broadcast算子中，设备1会将分区2的结果14直接写入设备3的分区2中。设备2会讲分区3的结果21直接写入设备1中。设备3会将分区1的结果直接写入设备2中。在一个3个节点的AllReduce集群中，我们会重复2次Broadcast算子来将每个分区的Reduce结果告知全部的节点。

### 带宽计算

在讨论集合通讯算子的性能时，人们经常会使用一些数值化指标去量化不同的算法实现，其中一个重要概念为带宽（Bandwidth）。在文献（:cite:`nvidia-nccl`）中，通常有两种主流的对带宽的计算方法，分别为算法带宽（Algorithm Bandwidth）与总线带宽（Bus Bandwidth）。

#### 算法带宽
前文提到，在计算点对点通信所需的时间是，会在信息长度之上乘以一个系数b。这个系数就是算法带宽，泛指单位时间内执行操作（通信，计算等）的数量。一般计算公式为b = s/t，其中s代指操作的大小，t指操作指定的两个端点之间所经过的时间。以点到点通信举例，我们可以通过衡量一个大小已知的信息m在执行send函数时所花的时间来确定两个处理单元之间网络的带宽。

#### 总线带宽
虽然算法带宽的计算方法既简单又高效，但很难将其拓展至对于集合通信算子的带宽计算。这是因为，取决于具体算子和算法实现的不同，一个集合通信算子在执行过程中测得的算法带宽往往会远小于硬件本身的最高带宽。在实际运行相应的测试中，经常能观测到随着处理单元增加，算法带宽呈下降趋势。为了解决这一问题，NCCL提出了总线带宽这一概念，通过对于每个集合通信算子的分析来对测得的算法带宽乘以一个校正系数（correction factor），来减轻处理单元数量对于测量带宽的影响并给出一个更贴近实际硬件表现的带宽值。下面列出了一些常见算子的校正系数，以及背后的简略推导。

* AllReduce：2(p-1)/p 对于在处理单元n_1, n_2 ... n_p 上的值 v_1, v_2 ... v_p 计算 v_1 op v_2 ... op v_p （其中op为符合结合律的算子），再存回每个处理单元中。在不考虑实际实现算法和网络拓扑的情况下，这个操作理论上只需要 2(p-1) 次数据传输，其中包含在每个处理单元上分开进行的 n-1 次 op的运算，以及最后 n 次最终数据值的广播，再减去第一个处理单元的运算和最后一个处理单元的广播的影响。假设每个处理单元对于外界所有信息处理的带宽为B，我们可以得出对于S个在不同处理单元上的数据运行AllReduce是能得到的最优情况下的运行时间：t = (2S(p-1)) / (pB)，进行简化后可得 B = (S/t)(2(p-1)/p) = b (2(p-1)/p)。这里的 2(p-1)/p便是我们的校正系数。
* ReduceScatter：(p-1)/p 对于每个处理单元来说，可以把ReduceScatter理解为只执行AllReduce中的聚合部分。对此，我们只需要考虑上文分析中的n-1次op的运算，整理后可得B = (S/t)((p-1)/p) = b ((p-1)/p)。
* AllGather：(p-1)/p 同理，对于每个处理单元来说，可以把AllGather理解为只执行AllReduce中的广播部分。我们同理可得B = (S/t)((p-1)/p) = b ((p-1)/p)。
* Broadcast：1 与AllReduce不同的是，Broadcast中所有数据需要从算子本身的发送者发出。即使在上文的分治情况下，我们也需要等待所有子问题运行结束才能确保Broadcast算子本身的正确性。因此，在计算带宽时瓶颈仍为发送者对于外界所有信息处理的带宽，所以 B = S/t，即校正系数为1。
* Reduce：1 同Broadcast，Reduce需要将所有数据送往算子的接受者，因此校正系数同样为1。

由于Gather和Scatter的带宽计算与实际聚合/分散时的数据结构相关性更高，故不给出特定的校正系数。
### 使用方法

针对不同的集群性质，现代机器学习系统往往会灵活应用不同集合通讯算子的组合来最大化通信效率。这里，我们提供了两个具体的案例分析，分别为微软的ZeRO 以及 OpenAI 的 DALL—E。

#### ZeRO
ZeRO （:cite:`rajbhandari2020zero`）是微软提出的神经网络优化器，可用于训练千亿级参数的神经网络，也在实践中成功训练了当时世界上最大的语言模型（为高达170亿参数的transformer）。在训练这个级别的神经网络时主要遇到的问题是巨量参数对于加速器内存的占用，其中包括优化器本身的参数，反向传播时的梯度，以及模型参数本身。通过简易的计算不难得出，170亿参数的模型在32位浮点表示情况下会占用至少680GB的内存，远超于现在内存最高的深度学习加速器A100 （最高内存80GB）。于是，我们需要考虑如何高效的把模型切成数份存储在不同的加速器上，以及如何高效的通过使用集合通信算子来进行模型训练和推理。ZeRO对此提出了多个优化方法，这里例举了三个典型的例子：
1. 首先，可以发现在现代集群中，节点内部加速器的带宽往往比节点之间的带宽要大很多。这在某种程度上偏离了上文中的理论框架。为此，我们需要尽量减少节点间的通信，尽量保证大部分通信仅存在于节点内部的加速器之间。在观察模型切分时，不难看出模型本身前馈和反向传播时需要大量的在不同切片之间通信，相比下来不同模型拷贝之间的梯度聚合反而具有相对较少的通信量。针对这一特性，ZeRO选择了将单一模型的全部切片存储到同一节点内部，从而大大提高了训练效率。
2. 进一步地，假设模型中的参数在层的细粒度上呈线性，便可将其从前到后分别存储到不同加速其中。在前馈时，可以注意到某一层的计算仅依赖于其相邻层的参数。对此，与其是手动设计点到点通信，我们可以对所有包含模型参数的加速器进行一次AllGather计算，用来提取每一层之后一层的参数，以及计算该层本身的激活值。为了节约内存，我们在AllGather结束后立即丢弃除了该层以外其他层的参数。
3. 同理，在反向传播时我们只需要前一层的参数来计算本层的激活值和梯度，因此我们只需要再次使用AllGather来完成每个加速器上的梯度计算。同时，我们注意到在聚集梯度后，对于每个加速器我们仅需要在内存中的层数的梯度。对此，我们可以使用ReduceScatter算子来在平均后直接把相应的梯度存到编号为i的加速器上，而不是通常情况下的AllReduce。

#### DALL-E
DALL-E （:cite:`ramesh2021zero`）是OpenAI提出的一个基于文字的图片生成模型，模型同样拥有高达120亿参数。在训练时，除了运用到ZeRO所使用的AllGather + ReduceScatter 技巧，OpenAI团队在细节上做了进一步的优化，以达到更快的训练速度。这里，我们简略介绍以下和集合通讯相关的两点：
1. 我们注意到，集合通讯算子的运行速度和通信本身的长度正相关。在模型训练中，这代表了模型参数本身的大小。对此，DALL-E 选择用矩阵分解（matrix factorization）的方法先把高维张量调整为一个二维矩阵，通过分解后分开用集合通信算子进行传输，从而大大减少了通信量。
2. 另一个减少通信量的方法在于数据类型本身。一个显然的做法是使用16位的半精度浮点数，相比正常的32位参数表示可以节省近一倍的通信量。但是，在实践中发现低精度的数据类型会使得模型收敛不稳定，往往导致最终训练效果大打折扣。为此，OpenAI分析了DALL—E 的模型结构，并把其中的参数根据对数据类型精度的敏感性分为了多个类。其中对精度最敏感的一类照常使用32位浮点表示并只通过AllReduce来同步，而最不敏感的参数则照常通过矩阵分解进行压缩和传输。对于比较敏感的一类，例如Adam 优化其中的动能（moments）和方差（variance）参数，OpenAI 基于 IEEE 754 标准实现了两个全新的数据类型：1-6-9和0-6-10（其中第一表示正负所需的位数，第二表示指数所需的位数，第三表示有效数字所需的位数），在节省空间和保持收敛性能之间找到了一个平衡。

### 集合通信与机器学习系统

最后，集合通信已经被深度集成到了整个机器学习系统之中，以至于一些在库级别以上的开发者很难意识到系统在训练和推理时的一些步骤是由底层逻辑实现的。接下来将介绍不同库对于集合通信的抽象程度，以及不同深度学习框架对于集合通信的支持程度和调用逻辑。

#### 集合通信的抽象
一般来说，不同的机器学习系统对于集合通信一般提供了两个级别的抽象，分别是更与硬件耦合的，可以直接调用集合通信算子的库，和更偏向神经网络实现的，通过内部调用集合通信算子来实现分布式训练和推理的深度学习框架。作为算法工程师，通常会接触到后者的抽象（包括Horovod, KungFu, TensorFlow distributed等），而作为集群的维护者，往往需要深入了解前者的运行原理和具体的调试方法。以深度学习框架 PyTorch 举例，在torch.distributed 命名空间（namespace）下实现了一系列方便开发者使用的分布式模型训练和推理函数。在其内部，会根据实际运行的集群调用更底层的集合通信算子库，例如MPI，NCCL（前文中已有介绍，适用于GPU分布式训练），gloo（适用于CPU分布式训练）等。我们来具体对比PyTorch distributed 中对于AllReduce 的应用和 NCCL 的差异性：下面两段代码中，前者（:cite:`li2022ddp`）通过PyTorch自带的分布式数据并行（Distributed Data Parallel）方法完成了一次简易的深度学习模型计算，后者则通过gloo的Python 接口pygloo和Ray（:cite:`moritz2018ray`）完成了一个二维张量的AllReduce计算。

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo")

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    run_demo(demo_basic, n_gpus)
```

```python
import os
import ray
import pygloo
import numpy as np
import multiprocessing

@ray.remote(num_cpus=1)
def test_allreduce(rank, world_size, fileStore_path):
    '''
    rank  # Rank of this process within list of participating processes
    world_size  # Number of participating processes
    fileStore_path # The path to create filestore
    '''
    context = pygloo.rendezvous.Context(rank, world_size)
    # Prepare device and store for rendezvous
    attr = pygloo.transport.tcp.attr("localhost")
    dev = pygloo.transport.tcp.CreateDevice(attr)
    fileStore = pygloo.rendezvous.FileStore(fileStore_path)
    store = pygloo.rendezvous.PrefixStore(str(world_size), fileStore)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data

    pygloo.allreduce(context, sendptr, recvptr,
                    sendbuf.size, pygloo.glooDataType_t.glooFloat32,
                    pygloo.ReduceOp.SUM, pygloo.allreduceAlgorithm.RING)

if __name__ == "__main__":
    ray.init()
    world_size = multiprocessing.cpu_count()
    fileStore_path = f"{ray.worker._global_node.get_session_dir_path()}" + "/collective/gloo/rendezvous"
    os.makedirs(fileStore_path)
    ray.get([test_allreduce.remote(rank, world_size, fileStore_path) for rank in range(world_size)])
```

可以注意到，前者并没有显示的调用集合通信算子，而是通过DistributedDataParallel将分布式训练和正常训练之间的不同隐藏了起来。如果我们需要在不同集群上运行这段代码，只需要在setup 函数内相对的更改PyTorch使用的底层集合通信库即可。在backward函数被调用时，才会真正的使用AllReduce算法。相比下来，如果想要直接使用gloo，不仅需要使用一步一步的创建通信所需要的数据结构，同时也很难和现有的模型训练框架无缝连接。

#### 通信接口
在讨论集合通信时，经常可以看到一些关于底层通信接口的专业术语，例如以太网，Infiniband 等。这里给出一些常见术语的简要解释，如有兴趣可以点击链接来进行拓展阅读。

* 以太网（Ethernet）：
* NVLink：
* AWS EFA：
* Infiniband：
* RDMA：
* RoCE：
* IPoIB：



