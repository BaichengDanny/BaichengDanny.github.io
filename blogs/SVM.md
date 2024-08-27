---
layout: page
permalink: /blogs/SVM/index.html
title: Support Vector Machine
---

# Support Vector Machine

> Baicheng Chen

支持向量机是一个经典的二分类模型，由Cortes和Vapnik于1995年提出。它区别于感知机，感知机是通过错误驱动的方式来确定一个可行的决策边界（minimize loss function => misclassification points，有**无穷多个**），而支持向量机则是选出“最优”的**一个**决策边界。这里的“最优”如何定义呢？SVM考虑了**最大间隔**（两个类别之间）。

这里，我们首先考虑一个简单的线性可分的二分类问题。

> 假设有样本点$\{(x_i, y_i)\}_i^N$, $x_i∈R^p$, $y∈\{-1,1\}$。
>
> <img src="https://gitee.com/baichengdanny/blogimage/raw/master/img/202408252217958.png" alt="image-20240825221715847" style="zoom: 50%;" />

现在，若我们想将绿色样本点与红色样本点分开，有无数条决策边界可以做到。下图中的1、2、3、4这四条线都可以完美的将这两种样本点进行分类。但我们可以发现1号线明显鲁棒性较差，如果样本点变多，它的泛化误差较大，并不能算一个足够“好”的边界。而我们需要的边界是一条鲁棒性好，可泛化的决策边界。

<img src="https://gitee.com/baichengdanny/blogimage/raw/master/img/202408252219572.png" alt="image-20240825221945519" style="zoom:50%;" />

于是，SVM变提出了“最大间隔”的概念。从上图中看，最大间隔便是两个类别最“中间”的那条线（即为3号线），它保持了到两个分类区域的距离最大，我们认为“最大间隔”的决策边界就是“最好”的决策边界。而这就引出了这篇文章的第一部分，也就是线性可分支持向量机，也称之为最大间隔分类器（Hard-margin SVM）。

## Hard-Margin SVM

现在，我们从数学的角度来定义“最大间隔”这个概念。首先，我们可以将决策边界看作一个超平面，用公式表示为$0=w^Tx+b$，这个超平面将$n$维空间分割为两半，其中，法向量$w$指向的那一半定义为正空间 ($∀x^+,w^Tx^++b>0$)，反之另一半则为负空间，可知：
$$
\max_{}\ margin(w,b)\\
s.t.\left\{
\begin{array}{ll}
w^Tx_i+b>0 & \text{, } y_i = +1 \\
w^Tx_i+b<0 & \text{, } y_i = -1
\end{array}
\right.
$$
这个公式还是不够简洁，我们尝试将constrains合并为一个条件，即：
$$
\max_{}\ margin(w,b)\\
s.t.\ \ y_i(w^Tx_i+b)>0\ ,\ \text{for } ∀i=1,2,...,N
$$
这个条件与上面两个条件是等价的，因为同号相乘一定为正（$y_i$与$w^Tx_i+b$始终同号）。

接下来，我们来详细推导$margin(w,b)$这个函数，即解释如何用数学形式来表示我们在前文所提到的“间隔”。简单来讲，“间隔”就是分类区域到决策边界的距离，在这里，我们认为N个样本点到决策边界的距离中最小的那个即为“间隔”。这样，如果我们通过最大化这个最小距离找到了一个最优的决策边界，它对于整个分类区域来说也一定是最优的。

假设这个最小距离由样本点$(x_i,y_i)$提供，下面计算$(x_i,y_i)$到决策边界$y=w^Tx+b$的距离：
$$
对于点\boldsymbol{x_0},设其在超平面0=\boldsymbol{w^Tx}+b的投影为\boldsymbol{x_1},则有\boldsymbol{w^Tx_1}+b=0。\\
因为法向量\boldsymbol{w}垂直于超平面,\boldsymbol{\vec{x_1x_0}}也垂直于超平面,故\boldsymbol{w}//\boldsymbol{\vec{x_1x_0}},则:\\
|\boldsymbol{w}·\boldsymbol{\vec{x_1x_0}}|=|||\boldsymbol{w}||·\cos{π}·||\boldsymbol{\vec{x_1x_0}}|||=||\boldsymbol{w}||·||\boldsymbol{\vec{x_1x_0}}||=||\boldsymbol{w}||·r,\ \ \ r为距离;\\
\begin{align}
\boldsymbol{w}·\boldsymbol{\vec{x_1x_0}}&=w_1(x_1^0-x_1^1)+w_2(x_2^0-x_2^1)+...+w_n(x_n^0-x_n^1)\\
&=w_{1}x_1^0+w_{2}x_2^0+...+w_{n}x_n^0-(w_{1}x_1^1+w_{2}x_2^1+...+w_{n}x_n^1)\\
&=\boldsymbol{w^Tx_0}-\boldsymbol{w^Tx_1}\\
&=\boldsymbol{w^Tx_0}+b \ \ \ \ \ \ \ \ (\because\  \boldsymbol{w^Tx_1}+b=0).
\end{align}
$$
所以有，
$$
\begin{align}
|\boldsymbol{w^Tx_0}+b|&=||\boldsymbol{w}·\boldsymbol{\vec{x_1x_0}}||\\
&=||\boldsymbol{w}||·r\\
⇒r&=\frac{|\boldsymbol{w^Tx_i}+b|}{||\boldsymbol{w}||}
\end{align}
$$
有了这个距离，我们就可以表示$margin(w,b)$函数了，注意：margin是样本点到决策边界的最小距离！
$$
\begin{align}
margin(w,b)&=\min_{w,b,x_i}{distance(w,b,x_i)}\\
&=\min_{w,b,x_i}\frac{|w^Tx_i+b|}{||w||}
\end{align}
$$
将导出的$margin(w,b)$公式带回到我们最开始写出的的优化问题中：
$$
\max_{w,b}\ \min_{x_i}\frac{|w^Tx_i+b|}{||w||}\\
s.t.\ \ \ y_i(w^Tx_i+b)>0
$$
因为我们有constrain $y_i(w^Tx_i+b)>0$，则可以将$y_i$看作$w^Tx_i+b$的绝对值符号，将其替换到objective function中：
$$
\max_{w,b}\ \min_{x_i}\frac{y_i(w^Tx_i+b)}{||w||}\\
s.t.\ \ \ y_i(w^Tx_i+b)>0
$$
因为$y_i(w^Tx_i+b)$始终大于0，我们假设$∃γ>0,s.t.\ \min_{x_i,y_i}{y_i(w^Tx_i+b)=γ}$。

则我们可以将objective function转化为：
$$
\begin{align}
\max_{w,b}\ \min_{x_i}\frac{y_i(w^Tx_i+b)}{||w||}&=\max_{w,b}\frac{1}{||w||} \min_{x_i}{y_i(w^Tx_i+b)}\\
&=\max_{w,b}\frac{1}{||w||}γ
\end{align}
$$
我们知道，超平面方程不唯一，即当我们等倍缩放$w$和$b$时，所得的新超平面与原超平面相同，故而，$γ$有无数种可能的取值。所以，我们添加一个约束，即令$γ=1$，以此来限制到唯一超平面。优化问题变成了：
$$
\max_{w,b}\ \frac{1}{||w||}\\
s.t.\ \ \min\ {y_i(w^Tx_i+b)=1}
$$
等价于，
$$
\max_{w,b}\ \frac{1}{||w||}\\
s.t.\ \ \ {y_i(w^Tx_i+b)≥1}
$$
等价于，
$$
\min_{w,b}\ {||w||}\\
s.t.\ \ \ {y_i(w^Tx_i+b)≥1}
$$
显然，这个优化问题可以被转化成一个QP问题，即：
$$
\min_{w,b}\ {\frac{1}{2}w^Tw}\\
s.t.\ \ \ {y_i(w^Tx_i+b)≥1},\ \ for\ \ ∀i=1,...,N
$$
等价于，
$$
\min_{w,b}\ {\frac{1}{2}w^Tw}\\
s.t.\ \ \ {1-y_i(w^Tx_i+b)≤0},\ \ for\ \ ∀i=1,...,N
$$
将这个QP问题看作一个约束优化问题，而上式则为该约束优化问题的原问题 (primal problem)。

下面，我们引入拉格朗日函数，
$$
L(w,b,\lambda)=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)), \ \ λ_i≥0
$$
通过拉格朗日函数的引入，我们可以找出原问题的无约束形式：
$$
\min_{w,b}\ \max_{λ}{L(w,b,λ)}\\
s.t.\ \ \ λ_i≥0
$$
下面对上述优化问题中的objective function进行解释：
$$
对于L(w,b,\lambda)=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)),\\
我们在左右两边同时取max,即:\\
\max_{λ}L(w,b,\lambda)=\frac{1}{2}w^Tw+\max_{λ}{(\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)))}\\
在该式中,λ_i≥0,1-y_i(w^Tx_i+b)≤0,我们可以得到,\\
\lambda_i(1-y_i(w^Tx_i+b))≤0\\
即\max_{λ}{(\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)))}=0\\
由此我们可以推出,\\
\begin{align}
\frac{1}{2}w^Tw&=\max_{λ}L(w,b,\lambda)-\max_{λ}{(\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)))}\\
&=\max_{λ}L(w,b,\lambda)-0\\
&=\max_{λ}L(w,b,\lambda)
\end{align}
$$
于是，我们就可以得出原问题的对偶问题 (dual problem)。因为该优化问题为凸二次优化问题，且约束条件为仿射函数，满足放松Slater条件，故原问题与对偶问题满足强对偶关系（convex + slater ⇒ strong duality）。
$$
\max_{λ}\ \min_{w,b}{L(w,b,λ)}\\
s.t.\ \ \ λ_i≥0
$$
下面，我们着手解决这个无约束优化问题。

- 对$b$求导，

$$
\begin{align}
\frac{\partial L}{\partial b}&=\frac{\partial}{\partial b}[\sum\limits_{i=1}^Nλ_i-\sum\limits_{i=1}^Nλ_iy_i(w^Tx_i+b)]\\
&=\frac{\partial}{\partial b}[-\sum\limits_{i=1}^Nλ_iy_ib]\\
&=-\sum\limits_{i=1}^Nλ_iy_i=0
\end{align}
$$

- 将上述结果带入原式化简，

$$
\begin{align}
L(w,b,λ)&=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b))\\
&=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i-\sum\limits_{i=1}^Nλ_iy_i(w^Tx_i+b)\\
&=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i-\sum\limits_{i=1}^Nλ_iy_iw^Tx_i-\sum\limits_{i=1}^Nλ_iy_ib\\
&=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i-\sum\limits_{i=1}^Nλ_iy_iw^Tx_i \ \ \ \ \ (\because \ -\sum\limits_{i=1}^Nλ_iy_i=0且b为常数)\\
\end{align}
$$

- 对$w$求导，

$$
\frac{\partial L}{\partial w}=\frac{1}{2}·2·w-\sum\limits_{i=1}^Nλ_iy_ix_i=0\\
\Rightarrow w^*=\sum\limits_{i=1}^Nλ_iy_ix_i
$$

- 将$w^*$代入objective function，

$$
\begin{align}
\min_{w,b}{L(w,b,λ)}&=\frac{1}{2}(\sum\limits_{i=1}^Nλ_iy_ix_i)^T(\sum\limits_{j=1}^Nλ_jy_jx_j)-\sum\limits_{i=1}^Nλ_iy_i(\sum\limits_{j=1}^Nλ_jy_jx_j)^Tx_i+\sum\limits_{i=1}^N\lambda_i\\
&=\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^Nλ_iλ
_jy_iy_jx_i^Tx_j-\sum\limits_{i=1}^N\sum\limits_{j=1}^Nλ_iλ
_jy_iy_jx_j^Tx_i+\sum\limits_{i=1}^N\lambda_i\\
&=-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^Nλ_iλ
_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i
\end{align}
$$

- 由上述推导，对偶问题可化为，

$$
\max_{λ}\ -\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^Nλ_iλ
_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i\\
s.t.\ \ \ λ_i≥0
$$

- 等价于，

$$
\min_{λ}\ \  \frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^Nλ_iλ
_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i\\
s.t.\ \ \ λ_i≥0
$$

因为原问题与对偶问题具有强对偶关系⇔满足KKT条件:
$$
KKT\left\{
\begin{array}{ll}
可行条件:\left\{
\begin{array}{ll}
λ_i≥0\\
1-y_i(w^Tx+b)≤0
\end{array}
\right.\\
互补松弛条件:λ_i(1-y_i(w^Tx+b))=0\\
梯度为0:\frac{\partial L}{\partial w}=0, \ \frac{\partial L}{\partial b}=0
\end{array}
\right.
$$
现在，我们再来看图示。中间蓝线便是我们要找出的最优决策边界$w^Tx+b=0$（线2），根据前文所述，我们将$γ$限制在1，故而，经过距离最优决策边界最近的两个样本点的直线可以分别表示为$w^Tx+b=1$（线1）和$w^Tx+b=-1$（线3）。

<img src="https://gitee.com/baichengdanny/blogimage/raw/master/img/202408261735480.png" alt="image-20240826173503281" style="zoom:50%;" />

以$w^Tx+b=1$为例（绿色区域），由可行条件$1-y_i(w^Tx+b)≤0$可知，此时为取等条件，即$1-y_i(w^Tx+b)=0$。由互补松弛条件可知，$λ$可取任意值，满足$λ_i≥0$。下面我们看到线1上方的区域，这里仍然存在四个绿色样本点，对于它们来说，$w^Tx+b>1$，所以$1-y_i(w^Tx+b)<0$，故而由互补松弛条件可知，$λ_i$一定等于零。所以，对于$λ_i$的值有贡献的样本点仅存在于$w^Tx+b=1$和$w^Tx+b=-1$两条线上，我们称这些点为**支持向量**。

下面，我们做最后的结果推导。根据KKT条件以及前文推导，我们已知：
$$
w^*=\sum\limits_{i=1}^Nλ_iy_ix_i
$$
从1和3两条线入手，进行$b^*$的推导：
$$
我们假设∃(x_k,y_k), \ s.t. \ 1-y_k(w^Tx_k+b)=0 ⇒ y_k(w^Tx_k+b)=1\\
因为y_k=±1,等式两边同乘y_k仍然成立,则:\\
y_k^2(w^Tx_k+b)=y_k\\
⇒b^*=y_k-w^Tx_k=y_k-\sum\limits_{i=1}^Nλ_iy_ix_i^Tx_k
$$
至此，我们便完成了Hard-margin SVM的全部推导，该判别模型最终可表示为一下分类决策函数：
$$
f(x)=sign(w^{*T}x+b^*)\\
w^*=\sum\limits_{i=1}^Nλ_iy_ix_i\\
b^*=y_k-w^Tx_k=y_k-\sum\limits_{i=1}^Nλ_iy_ix_i^Tx_k
$$

## Soft-Margin SVM

在引入部分我们便给Hard-margin SVM添加了一个限制条件，即我们的二分类问题一定是线性可分的。那么如果我们拿到的二分类问题是线性不可分问题呢？这里，我们就引入了Soft-margin SVM，它相当于在线性可分问题的基础上允许一些错误样本点（整体分类问题是线性不可分的）。

回顾我们在推导Hard-margin SVM时的原问题，即：
$$
\min_{w,b}\ {\frac{1}{2}w^Tw}\\
s.t.\ \ \ {1-y_i(w^Tx_i+b)≤0},\ \ for\ \ ∀i=1,...,N
$$
Soft-margin SVM的基本思想是引入一个loss function用来表示错误分类损失，并将其一起最小化，获得尽可能优的决策边界，我们可以写作：
$$
\min_{w,b}\ {\frac{1}{2}w^Tw+loss\  function}\\
s.t.\ \ \ {1-y_i(w^Tx_i+b)≤0},\ \ for\ \ ∀i=1,...,N
$$
下面我们来探讨合适的loss function。

首先，最为简单的loss function便是错分的样本点数量，数学公式可以写作：
$$
loss=\sum\limits_{i=1}^NI\{y_i(w^Tx_i+b)<1\}
$$
但此时的问题是这个函数并不连续，这样的优化问题很难求解，我们选择考虑别的loss function。

既然数量不行，距离行不行呢？这里，我们便引入了Hinge Loss。我们做如下考虑，假设我们已分类了样本点$(x_i,y_i)$，如果该样本点满足约束条件（函数间隔（确信度）），即$y_i(w^Tx_i+b)≥1$，令$loss=0$；相反，如果该样本点不满足约束条件，即$y_i(w^Tx_i+b)<1$，令$loss = 1-y_i(w^Tx_i+b)$。

合并成一个连续的loss function，
$$
loss = max\{0,1-y_i(w^Tx_i+b)\}
$$
设$z=y_i(w^Tx_i+b)$， $loss = max\{0,1-z\}$，画出图像如下（合页形状）：

<img src="https://gitee.com/baichengdanny/blogimage/raw/master/img/202408262046238.png" alt="image-20240826204632613" style="zoom:50%;" />

我们发现，这个损失函数是连续可微的，可以使用，于是写出Soft-margin SVM的优化问题如下：
$$
\min_{w,b}\ {\frac{1}{2}w^Tw+C\sum\limits_{i=1}^N\max\{0,1-y_i(w^Tx_i+b)\}}\\
s.t.\ \ \ {y_i(w^Tx_i+b)≥1},\ \ for\ \ ∀i=1,...,N
$$
这里，$C>0$是一个超参数，我们称为惩罚参数，由实际应用问题决定。C越大，对错误分离的惩罚越大。

为进一步消除$max$函数，我们引入一个松弛变量$\xi_i=1-y_i(w^Tx_i+b),\ \ \xi_i≥0$。上式可简化为，
$$
\min_{w,b}\ {\frac{1}{2}w^Tw+C\sum\limits_{i=1}^N\xi_i}\\
s.t.\ \ \ {y_i(w^Tx_i+b)≥1},\ \ for\ \ ∀i=1,...,N\\
\xi_i≥0\ \ \ \ \ \ 
$$
接下来，使用约束优化问题的方法同Hard-margin SVM一样正常推导即可。

## Kernel Method

前文中，我们讨论了对于线性可分二分类问题、线性不可分二分类问题的解决方法，现在，如果二分类问题变成了非线性问题，我们该怎么解决呢？对于这种问题，我们便要引入核方法。因篇幅有限，我会在下一篇文章详细讨论核方法的原理、核函数及正定核等内容。

