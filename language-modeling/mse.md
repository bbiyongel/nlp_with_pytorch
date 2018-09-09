# Neural Network equals to Probability Function

## KL Divergence

$$
\begin{aligned}
KL(P||P_\theta)&=-\mathbb{E}_{X \sim P}[\log{\frac{P_\theta(X)}{P(X)}}] \\
&=-\sum_{x\in\mathcal{X}}{P(x)\log{\frac{P_\theta(x)}{P(x)}}} \\
&=-\sum_{x\in\mathcal{X}}{\Big(P(x)\log{P_\theta(x)}-P(x)\log{P(x)}\Big)} \\
&=H(P,P_\theta)-H(P)
\end{aligned}
$$

$$
\begin{aligned}
\nabla_\theta KL(P||P_\theta)&=\nabla_\theta\big(H(P,P_\theta)-H(P)\big) \\
&=\nabla_\theta H(P,P_\theta)
\end{aligned}
$$

## Cross Entropy Loss

An objective function by Cross Entropy is

$$
\begin{aligned}
J(\theta)=H(P,P_\theta)&=-\mathbb{E}_{X\sim P(X)}\Big[\mathbb{E}_{Y\sim P(Y|X)}[\log{P(Y|X;\theta)}]\Big] \\
&=-\sum_{x\in\mathcal{X}}{P(x)\sum_{y\in\mathcal{Y}}{P(y|x)\log{P(y|x;\theta)}}} \\
\end{aligned}
$$

Assume that we have a data set with $N$ samples,

$$
\mathcal{B}=\{x,y\}_{i=1}^N
$$

By Monte-Carlo Sampling,

$$
\begin{aligned}
J(\theta)&\approx-\frac{1}{N}\sum_{i=1}^N{\frac{1}{K}\sum_{j=1}^K{\log{P(y_j|x_i;\theta)}}} \\
&\approx-\frac{1}{N}\sum_{i=1}^N{\log{P(y_i|x_i;\theta)}}
\end{aligned}
$$

Note that $K=1$, in our data set. 

To minimize the objective function,

$$
\begin{gathered}
\hat{\theta}=\underset{\theta}{\text{argmin }}J(\theta) \\ \\
\theta \leftarrow \theta-\lambda\nabla_\theta J(\theta)
\end{gathered}
$$

## Negative Log-likelihood Function

## Mean Square Error (MSE) Loss

Unlike discrete variable, we cannot get probability from continuous variable. In this case, we consider a data point is sampled from a gaussian distribution. Probability density function (PDF) for gaussian distribution is

$$
\mathcal{N}(\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}\exp{(-\frac{(x-\mu)^2}{2\sigma^2})}
$$

Thus, our neural network can be treated like a probability function with continuous variable, and it returns a gaussian distribution.

$$
f_\theta(x)=\mathcal{N}\big(\mu_\phi(x), \sigma_\psi(x)^2\big)
$$

Now, we can write an objective function using log-likelihood function. Note that log-likelihoood for continuous function conists of probability density function (PDF). In many cases, we do not consider a standard deviation. So, out objective function can be written like as below:

$$
\begin{aligned}
J(\theta)=-\frac{1}{N}\sum_{i=1}^N{\log{f_\theta(x_i)}}
&=-\frac{1}{N}\sum_{i=1}^N{\log{\Big(\frac{1}{\sigma_\psi(x_i)\sqrt{2\pi}}\exp{(-\frac{\big(x_i-\mu_\phi(x_i)\big)^2}{2\sigma_\psi(x_i)^2})}\Big)}}, \\
&=-\frac{1}{N}\sum_{i=1}^N\Big(\log{\frac{1}{\sigma_\psi(x_i)\sqrt{2\pi}}}-\frac{\big(x_i-\mu_\phi(x_i)\big)^2}{2\sigma_\psi(x_i)^2}\Big) \\
&=\log{\sigma_\psi(x_i)}+\frac{1}{2}\log{2\pi}+\frac{1}{2\sigma_\psi(x_i)^2\cdot N}\sum_{i=1}^N\big(x_i-\mu_\phi(x_i)\big)^2\\
&\text{where }\theta^*=\{\phi,\psi\}\text{, but ignore }\psi\text{ and }\theta=\{\phi\}.
\end{aligned}
$$

Also, we need to minimize this objective function to estimate the parameter, $\theta$. $\theta$ can be achieved by gradient descent.

$$
\begin{gathered}
\hat{\theta}=\underset{\theta}{\text{argmin }}J(\theta) \\ \\
\theta \leftarrow \theta-\lambda\nabla_\theta J(\theta)
\end{gathered}
$$

If we take a derivative of objective function, many terms can be removed because there are constant.

$$
\nabla_\theta{J(\theta)}=\nabla_\theta{\frac{1}{2\sigma^2\cdot N}\sum_{i=1}^N\big(x_i-\mu_\phi(x_i)\big)^2}
$$

Therefore, we can re-write the objective function, which is very similar form.

$$
\tilde{J}(\theta)=\frac{1}{N}\sum_{i=1}^N\big(x_i-\mu_\phi(x_i)\big)^2
$$