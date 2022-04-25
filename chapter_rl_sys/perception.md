## 感知系统

感知系统不仅可以包括视觉，还可以包含触觉、声音等。在未知环境中，机器人想实现自主移动和导航必须知道自己在哪（例如通过相机重定位 :cite:`ding2019camnet`），周围什么情况（例如通过3D物体检测 :cite:`yi2020segvoxelnet`或语义分割），这些要依靠感知系统来实现 :cite:`xu2019depth` :cite:`xu2020selfvoxelo` :cite:`xu2022rnnpose` :cite:`xu2022robust` :cite:`yang2021pdnet` :cite:`huang2021vs` :cite:`huang2021life:cite:` :cite:`huang2019prior` :cite:`zhu2020ssn`。
一提到感知系统，不得不提的就是即时定位与建图（Simultaneous Localization
and
Mapping，SLAM)系统。SLAM大致过程包括地标提取、数据关联、状态估计、状态更新以及地标更新等。视觉里程计Visual
Odometry是SLAM中的重要部分，它估计两个时刻机器人的相对运动（Ego-motion）。ORB-SLAM :cite:`campos2021orb`系列是视觉SLAM中有代表性的工作， :numref:`orbslam3` 展示了最新的ORB-SLAM3的主要系统组件。香港科技大学开源的基于单目视觉与惯导融合的SLAM技术VINS-Mono :cite:`8421746`也很值得关注。多传感器融合、优化数据关联与回环检测、与前端异构处理器集成、提升鲁棒性和重定位精度都是SLAM技术接下来的发展方向。

最近，随着机器学习的兴起，基于学习的SLAM框架也被提了出来。TartanVO :cite:`tartanvo2020corl`是第一个基于学习的视觉里程计（VO）模型，该模型可以推广到多个数据集和现实世界场景，并优于传统基于几何的方法。
UnDeepVO :cite:`li2018undeepvo`是一个无监督深度学习方案，能够通过使用深度神经网络估计单目相机的
6-DoF 位姿及其视图深度。DROID-SLAM :cite:`teed2021droid`是用于单目、立体和
RGB-D 相机的深度视觉 SLAM，它通过Bundle
Adjustment层对相机位姿和像素深度的反复迭代更新，具有很强的鲁棒性，故障大大减少，尽管对单目视频进行了训练，但它可以利用立体声或
RGB-D 视频在测试时提高性能。其中，Bundle Adjustment
(BA)与机器学习的结合被广泛研究 :cite:`tang2018ba` :cite:`tanaka2021learning`。CMU提出通过主动神经
SLAM
的模块化系统帮助智能机器人在未知环境中的高效探索 :cite:`chaplot2020learning`。

![ORB-SLAM3主要系统组件 :cite:`campos2021orb`](../img/ch13/orbslam3.png)

:width:`800px`

:label:`orbslam3`
