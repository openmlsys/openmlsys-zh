## 梯度下降与反向传播

上面大体上介绍了经典神经网络的内容，那么现在有一个问题，这些网络中的参数是如何确定的呢？如果要解决的问题是一个小感知器就能解决的话，参数可以人为地去确定。但是如果是一个深度网络的话，参数的确定需要自动化，也就是所谓的网络训练，而这个过程需要我们设定一个**损失函数**（Loss
Function）来作为训练优化的一个方向。
常见的损失函数有：1）用来衡量向量之间距离的均方误差(Mean Squared
Error，MSE)
$\mathcal{L} = \frac{1}{N}\|\bm{y}-\hat{\bm{y}}\|^{2}_{2} = \frac{1}{N}\sum_{i=1}^N(y_{i}-\hat{y}_{i})^{2}$
和 平均绝对误差(Mean Absolute Error，MAE)
$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}|y_{i}-\hat{y}_{i}|$
，其中$N$代表数据样本的数量，用以求平均用，而$y$代表真实标签（Ground
Truth）、$\hat{y}$代表网络输出的预测标签。
2）分类任务可以用的交叉熵损失（Cross Entropy）
$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \bigg(y_{i}\log\hat{y}_{i} + (1 - y_{i})\log(1 - \hat{y}_{i})\bigg)$来作为损失数，当且仅当输出标签和预测标签一样的时候损失值才为零。

有了损失值之后，我们就可以利用大量真实标签的数据和优化方法来更新模型参数了，其中最常用的方法是**梯度下降**（Gradient
Descent）。如 :numref:`gradient_descent2`所示，
开始的时候，模型的参数$\bm{w}$是随机选取的，然后求出损失值对参数的偏导数$\frac{\partial \mathcal{L}}{\partial \bm{w}}$，通过反复迭代
$\bm{w}:=\bm{w}-\alpha\frac{\partial \mathcal{L}}{\partial \bm{w}}$完成优化。这个优化的过程其实就可以降低损失值以达到任务目标，其中$\alpha$是控制优化幅度的**学习率**（Learning
Rate）。
在实践中，梯度下降最终得到的最小值很大可能是一个局部最小值，而不是全局最小值。不过由于深度神经网络能提供一个很强的数据表达能力，所以局部最小值可以很接近全局最小值，损失值可以足够小。

![梯度下降介绍。（左图）只有一个可以训练的参数$w$；（右图）有两个可以训练的参数$\bm{w}=[w_1,w_2]$。在不断更新迭代参数后，损失值$\mathcal{L}$会逐渐地减小。但是由于存在很多局部最优解，我们往往不能更新到全局最优解。](../img/ch_basic/gradient_descent2.png)
:width:`600px`
:label:`gradient_descent2`

那么接下来，在深度神经网络中如何实现梯度下降呢，这需要计算出网络中每层参数的偏导数$\frac{\partial \mathcal{L}}{\partial \bm{w}}$，我们可以用**反向传播**（Back-Propagation）[@rumelhart1986learning; @lecun2015deep]来实现。
接下来，
我们引入一个中间量$\bm{\delta}=\frac{\partial \mathcal{L}}{\partial \bm{z}}$来表示损失函数$\mathcal{L}$
对于神经网络输出$\bm{z}$（未经过激活函数，不是$a$）的偏导数，
并最终得到$\frac{\partial \mathcal{L}}{\partial \bm{w}}$。

我们下面用一个例子来介绍反向传播算法，
我们设层序号为$l=1, 2, \ldots  L$（输出层（最后一层）序号为$L$）。
对于每个网络层，我们有输出$\bm{z}^l$，中间值$\bm{\delta}^l=\frac{\partial \mathcal{L}}{\partial \bm{z}^l}$和一个激活值输出$\bm{a}^l=f(\bm{z}^l)$
（其中$f$为激活函数）。
我们假设模型是使用Sigmoid激活函数的多层感知器，损失函数是均方误差（MSE）。也就是说，我们设定：

-   网络结构$\bm{z}^{l}=\bm{W}^{l}\bm{a}^{l-1}+\bm{b}^{l}$

-   激活函数$\bm{a}^l=f(\bm{z}^l)=\frac{1}{1+{\rm e}^{-\bm{z}^l}}$

-   损失函数$\mathcal{L}=\frac{1}{2}\|\bm{y}-\bm{a}^{L}\|^2_2$

我们可以直接算出激活输出对于原输出的偏导数：

-   $\frac{\partial \bm{a}^l}{\partial \bm{z}^l}=f'(\bm{z}^l)=f(\bm{z}^l)(1-f(\bm{z}^l))=\bm{a}^l(1-\bm{a}^l)$

和损失函数对于激活输出的偏导数：

-   $\frac{\partial \mathcal{L}}{\partial \bm{a}^{L}}=(\bm{a}^{L}-\bm{y})$

有了这些后，为了进一步得到损失函数对于每一个参数的偏导数，可以使用**链式法则**（Chain
Rule），细节如下：

首先，从输出层（$l=L$，最后一层）开始向后方传播误差，根据链式法则，我们先计算输出层的中间量：

-   $\bm{\delta}^{L}
    =\frac{\partial \mathcal{L}}{\partial \bm{z}^{L}}
    =\frac{\partial \mathcal{L}}{\partial \bm{a}^{L}}\frac{\partial \bm{a}^L}{\partial \bm{z}^{L}}=(\bm{a}^L-\bm{y})\odot(\bm{a}^L(1-\bm{a}^L))$

除了输出层（$l=L$）的中间值$\bm{\delta}^{L}$，其他层（$l=1, 2, \ldots , L-1$）的中间值$\bm{\delta}^{l}$如何计算呢？

-   已知模型结构$\bm{z}^{l+1}=\bm{W}^{l+1}\bm{a}^{l}+\bm{b}^{l+1}$，我们可以直接得到$\frac{\partial \bm{z}^{l+1}}{\partial \bm{a}^{l}}=\bm{W}^{l+1}$；而且我们已知$\frac{\partial \bm{a}^l}{\partial \bm{z}^l}=\bm{a}^l(1-\bm{a}^l)$

-   那么根据链式法则，我们可以得到 $\bm{\delta}^{l}
    =\frac{\partial \mathcal{L}}{\partial \bm{z}^{l}}
    =\frac{\partial \mathcal{L}}{\partial \bm{z}^{l+1}}\frac{\partial \bm{z}^{l+1}}{\partial \bm{a}^{l}}\frac{\partial \bm{a}^{l}}{\partial \bm{z}^{l}}
    =(\bm{W}^{l+1})^\top\bm{\delta}^{l+1}\odot(\bm{a}^l(1-\bm{a}^l))$

根据上面的计算有所有层的中间值$\bm{\delta}^l, l=1, 2, \ldots , L$后，我们就可以在此基础上求出损失函数对于每层参数的偏导数：$\frac{\partial \mathcal{L}}{\partial \bm{W}^l}$和$\frac{\partial \mathcal{L}}{\partial \bm{b}^l}$，以此来根据梯度下降的方法来更新每一层的参数。

-   已知模型结构$\bm{z}^l=\bm{W}^l\bm{a}^{l-1}+\bm{b}^l$，我们可以求出
    $\frac{\partial \bm{z}^{l}}{\partial \bm{W}^l}=\bm{a}^{l-1}$ 和
    $\frac{\partial \bm{z}^{l}}{\partial \bm{b}^l}=1$

-   那么根据链式法则，我们可以得到$\frac{\partial \mathcal{L}}{\partial \bm{W}^l}=\frac{\partial \mathcal{L}}{\partial \bm{z}^l}\frac{\partial \bm{z}^l}{\partial \bm{W}^l}=\bm{\delta}^l(\bm{a}^{l-1})^\top$
    ,
    $\frac{\partial \mathcal{L}}{\partial \bm{b}^l}=\frac{\partial \mathcal{L}}{\partial \bm{z}^l}\frac{\partial \bm{z}^l}{\partial \bm{b}^l}=\bm{\delta}^l$

求得所有偏导数$\frac{\partial \mathcal{L}}{\partial \bm{W}^l}$ 和
$\frac{\partial \mathcal{L}}{\partial \bm{b}^l}$后，我们就可以用梯度下降更新所有参数$\bm{W}^l$
和 $\bm{b}^l$：

-   $\bm{W}^l:=\bm{W}^l-\alpha\frac{\partial \mathcal{L}}{\partial \bm{W}^l}$,
    $\bm{b}^l:=\bm{b}^l-\alpha\frac{\partial \mathcal{L}}{\partial \bm{b}^l}$

但是还有一个问题需要解决，那就是梯度下降的时候每更新一次参数，都需要计算一次当前参数下的损失值。然而，当训练数据集很大时（$N$很大），若每次更新都用整个训练集来计算损失值的话，计算量会非常巨大。
为了减少计算量，我们使用**随机梯度下降**（Stochastic Gradient
Descent，SGD）来计算损失值。具体来说，我们计算损失值不用全部训练数据，而是从训练集中随机选取一些数据样本来计算损失值，比如选取16、32、64或者128个数据样本，样本的数量被称为**批大小**（Batch
Size）。
此外，学习率的设定也非常重要。如果学习率太大，可能无法接近最小值的山谷，如果太小，训练又太慢。
自适应学习率，例如Adam [@KingmaAdam2014]、RMSProp [@tieleman2012rmsprop]
和
Adagrad [@duchi2011adagrad] 等，在训练的过程中通过自动的方法来修改学习率，实现训练的快速收敛，到达最小值点。
