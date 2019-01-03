# Temporal Note for Markdown Equations

## Add one Smoothing

$$\begin{aligned}
P(w_i|w_{<i}) \approx \frac{C(w_{<i},w_i)+1}{C(w_{<i})+V}
\end{aligned}$$

$$\begin{aligned}
P(w_i|w_{<i}) &\approx \frac{C(w_{<i},w_i)+k}{C(w_{<i})+kV} \\
&\approx \frac{C(w_{<i},w_i)+(m/V)}{C(w_{<i})+m}
\end{aligned}$$

$$P(w_i|w_{<i}) \approx \frac{C(w_{<i},w_i)+m P(w_i)}{C(w_{<i})+m}$$

## Auto-regressive and Teacher Forcing on RNNLM

$$X=argmax_X P(X)=argmax_Y \prod_{i=1}^{n}{P(x_i|x_{<i})}$$

$$\begin{aligned}
x_i=&argmax_y{P(x|x_{<i})} \\
&\text{where }x_0=BOS
\end{aligned}$$

$$\hat{x}_t=argmax_x{P(x_t|x_{<t};\theta)}\text{ where }X=\{x_1,x_2,\cdots,x_n\}$$

$$\begin{aligned}
\mathcal{L}(X)&=-\sum_{t=1}^{n+1}{\log{P(x_t|x_{<t};\theta)}} \\
\theta &\leftarrow \theta-\lambda\frac{1}{N}\sum_{i=1}^{N}{\mathcal{L}(X_i)}
\end{aligned}$$

## Expectation

$$\begin{aligned}
\mathbb{E}_{x \sim p}[reward(x)]&=\int{reward(x)p(x)}dx \\
&\approx\frac{1}{K}\sum_{i=1}^{K}{reward(x_i)} \\
&\approx reward(x)
\end{aligned}$$

## Probability

### Discrete Variable

$$P(X=x)$$

### Continuous Variable

$$\forall x \in X,~p(x)\ge0.$$

$$\text{We don not require that }p(x)\le1.$$

$$\int_{-\infty}^{\infty}{p(x)}~dx=1$$

### Conditional Probability

$$P(A|B)=\frac{P(A,B)}{P(B)}$$

#### Conditional Independence

$$P(A,B)=P(A)P(B)$$

$$P(A|B)=\frac{P(A,B)}{P(B)}=P(A)$$

### Marginal Probability

$$P(y)=\sum_{x\in\mathcal{X}}{P(y,x)}=\sum_{x\in\mathcal{X}}{P(y|x)P(x)}$$

### Bayes Theorem

$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$

### Bernoulli Distribution

$$P(x=1|\mu)=\mu$$

### Gaussian Distribution

$$\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{1/2}}\exp{\Big\{-\frac{1}{2\sigma^2}(x-\mu)^2\Big\}}$$

## Machine Learning

### MLE

$$\hat{\theta}=\underset{\theta}{\text{argmax }}P(Y|X;\theta)=\underset{\theta}{\text{argmax }}P(Y|X,\theta)$$

$$\hat{\theta}=\underset{\theta}{\text{argmax }}P(X;\theta)=\underset{\theta}{\text{argmax }}P(X|\theta)$$

#### Example



### MAP

$$\begin{aligned}
\hat{\theta}&=\underset{\theta}{\text{argmax }}P(\theta|X,Y) \\
&=\underset{\theta}{\text{argmax }}P(X,Y,\theta) \\
&=\underset{\theta}{\text{argmax }}P(Y|X;\theta)P(X,\theta) \\
&=\underset{\theta}{\text{argmax }}P(Y|X;\theta)P(X;\theta)P(\theta)
\end{aligned}$$

$$\begin{aligned}
\hat{\theta}&=\underset{\theta}{\text{argmax }}P(\theta|X) \\
&=\underset{\theta}{\text{argmax }}P(X,\theta) \\
&=\underset{\theta}{\text{argmax }}P(X;\theta)P(\theta)
\end{aligned}$$

### Ensemble

$$P(Y|X)=\mathbb{E}_{\theta\sim P}[P(Y|X;\theta)]\approx\frac{1}{N}\sum_{i=1}^N{P(Y|X;\theta)}$$

## Joint Training for Neural Machine Translation Models with Monolingual Data

$$\mathcal{L}^*(\theta_{x\rightarrow y})=\sum_{n=1}^N{\log{p(y^{(n)}|x^{(n)})}}+\sum_{t=1}^T{\log{p(y^{(t)})}}$$

$$\begin{aligned}
\log{p(y^{(t)})}&=\log{\sum_x{p(x,y^{(t)})}}=\log{\sum_x{Q(x)\frac{p(x,y^{(t)})}{Q(x)}}} \\
&\ge\sum_x{Q(x)\log{\frac{p(x,y^{(t)})}{Q(x)}}} \\
&=\sum_x{\Big(Q(x)\log{p(y^{(t)}|x)-\text{KL}(Q(x)||p(x))}\Big)}
\end{aligned}$$

$$\frac{p(x,y^{(t)})}{Q(x)}=c$$

$$Q(x)=\frac{p(x,y^{(t)})}{c}=\frac{p(x,y^{(t)})}{\sum_x{p(x,y^{(t)})}}=p^*(x|y^{(t)})$$

$$\mathcal{L}^*(\theta_{x\rightarrow y})\ge\mathcal{L}(\theta_{x\rightarrow y})=\sum_{n=1}^N{\log{p(y^{(n)}|x^{(n)})}}+\sum_{t=1}^T{\sum_x{\Big(p(x|y^{(t)})\log{p(y^{(t)}|x)}-\text{KL}\big(p(x|y^{(t)})||p(x)\big)\Big)}}$$

$$\mathcal{L}(\theta_{x\rightarrow y})=\sum_{n=1}^N{\log{p(y^{(n)}|x^{(n)})}}+\sum_{t=1}^T{\sum_x{p(x|y^{(t)})\log{p(y^{(t)}|x)}}}$$

### Re-write

$$\mathcal{L}(\theta)=-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\sum_{s=1}^S{\log{P(y^s)}}$$

$$\begin{aligned}
\log{P(y)}&=\log{\sum_{x\in\mathcal{X}}{P(y|x)P(x)}} \\
&=\log{\sum_{x\in\mathcal{X}}{P(x|y)\frac{P(y|x)P(x)}{P(x|y)}}} \\
&\ge\sum_{x\in\mathcal{X}}{P(x|y)\log{\frac{P(y|x)P(x)}{P(x|y)}}} \\
&=\mathbb{E}_{x\sim P(x|y)}[\log{P(y|x)}+\log{\frac{P(x)}{P(x|y)}}] \\
&=\mathbb{E}_{x\sim P(x|y)}[\log{P(y|x)}]+\mathbb{E}_{x\sim P(x|y)}[\log{\frac{P(x)}{P(x|y)}}] \\
&=\mathbb{E}_{x\sim P(x|y)}[\log{P(y|x)}]-\text{KL}\big(P(x|y)||P(x)\big)
\end{aligned}$$

$$-\log{P(y)}\le-\mathbb{E}_{x\sim P(x|y)}[\log{P(y|x)}]+\text{KL}\big(P(x|y)||P(x)\big)$$

$$\begin{aligned}
\mathcal{L}(\theta)&\le-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\sum_{s=1}^S{\Big(\mathbb{E}_{x\sim P(x|y^s)}[\log{P(y^s|x;\theta)}]-\text{KL}\big(P(x|y^s)||P(x)\big)\Big)} \\
&\approx-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\frac{1}{K}\sum_{s=1}^S{\sum_{i=1}^K{\log{P(y^s|x_i;\theta)}}}+\sum_{s=1}^S{\text{KL}\big(P(x|y^s)||P(x)\big)} \\
&=\tilde{\mathcal{L}}(\theta)
\end{aligned}$$

$$\nabla_\theta\tilde{\mathcal{L}}(\theta)=-\sum_{n=1}^N{\nabla_\theta\log{P(y^n|x^n;\theta)}}-\frac{1}{K}\sum_{s=1}^S{\sum_{i=1}^K{\nabla_\theta\log{P(y^s|x_i;\theta)}}}$$

<!--
### Upgrade

$$\mathcal{L}(\theta,\phi)=-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\sum_{n=1}^N{\log{P(x^{n}|y^{n};\phi)}}-\sum_{s=1}^S{\log{P(x^s)}}-\sum_{t=1}^T{\log{P(y^t)}}$$

$$\begin{aligned}
\mathcal{L}(\theta,\phi)&\le-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\sum_{n=1}^N{\log{P(x^{n}|y^{n};\phi)}}-\sum_{s=1}^S{\Big(\mathbb{E}_{y\sim P(y|x^s)}[\log{P(x^s|y;\phi)}]-\text{KL}\big(P(y|x^s;\theta)||P(y)\big)\Big)}-\sum_{t=1}^T{\Big(\mathbb{E}_{x\sim P(x|y^t)}[\log{P(y^t|x;\theta)}]-\text{KL}\big(P(x|y^t;\phi)||P(x)\big)\Big)} \\
&=\Big(-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\sum_{t=1}^T{\sum_{x\in\mathcal{X}}{P(x|y^t;\phi)\cdot\log{P(y^t|x;\theta)}}}+\sum_{s=1}^S{\text{KL}\big(P(y|x^s;\theta)||P(y)\big)}\Big)+\Big(-\sum_{n=1}^N{\log{P(x^{n}|y^{n};\phi)}}-\sum_{s=1}^S{\sum_{y\in\mathcal{Y}}{P(y|x^s;\theta)\cdot\log{P(x^s|y;\phi)}}}+\sum_{t=1}^T{\text{KL}\big(P(x|y^t;\phi)||P(x)\big)}\Big) \\
&=\mathcal{L}(\theta)+\mathcal{L}(\phi) \\
&\approx\Big(-\sum_{n=1}^N{\log{P(y^{n}|x^{n};\theta)}}-\frac{1}{K}\sum_{t=1}^T{\sum_{i=1}^K{\log{P(y^t|x_i;\theta)}}}+\sum_{s=1}^S{\text{KL}\big(P(y|x^s;\theta)||P(y)\big)}\Big)+\Big(-\sum_{n=1}^N{\log{P(x^{n}|y^{n};\phi)}}-\frac{1}{K}\sum_{s=1}^S{\sum_{i=1}^K{\log{P(x^s|y_i;\phi)}}}+\sum_{t=1}^T{\text{KL}\big(P(x|y^t;\phi)||P(x)\big)}\Big) \\
&=\tilde{\mathcal{L}}(\theta)+\tilde{\mathcal{L}}(\phi)=\tilde{\mathcal{L}}(\theta,\phi)
\end{aligned}$$

$$\begin{aligned}
\nabla_\theta J(\theta,\phi)&=\sum_{y\in\mathcal{Y}}{\nabla_\theta P(y|x;\theta)\cdot\log{P(x|y;\phi)}} \\
&=\sum_{y\in\mathcal{Y}}{P(y|x;\theta)\nabla_\theta\log{P(y|x;\theta)}\cdot\log{P(x|y;\phi)}} \\
&=\mathbb{E}_{y\sim P(y|x;\theta)}[\nabla_\theta\log{P(y|x;\theta)}\log{P(x|y;\phi)}] \\
&\approx\frac{1}{K}\sum_{i=1}^K{\log{P(x|y_i;\phi)\nabla_\theta\log{P(y_i|x;\theta)}}}
\end{aligned}$$

$$\begin{aligned}
\nabla_\theta\mathcal{L}(\theta,\phi)&=\nabla_\theta\mathcal{L}(\theta)+\nabla_\phi\mathcal{L}(\phi) \\
&\approx\nabla_\theta\tilde{\mathcal{L}}(\theta)-\nabla_\theta J(\theta,\phi) \\
&=\nabla_\theta\tilde{\mathcal{L}}(\theta)-\frac{1}{K}\sum_{s=1}^S{\sum_{i=1}^K{\log{P(x^s|y_i;\phi)\nabla_\theta\log{P(y_i|x^s;\theta)}}}}
\end{aligned}$$

$$\nabla_\theta\tilde{\mathcal{L}}(\theta,\phi)=\nabla_\theta\tilde{\mathcal{L}}(\theta)+\nabla_\theta\tilde{\mathcal{L}}(\phi)=\nabla_\theta\tilde{\mathcal{L}}(\theta)$$

$$\nabla_\phi\tilde{\mathcal{L}}(\theta,\phi)=\nabla_\phi\tilde{\mathcal{L}}(\theta)+\nabla_\phi\tilde{\mathcal{L}}(\phi)=\nabla_\phi\tilde{\mathcal{L}}(\phi)$$
-->

### DUL

$$\begin{aligned}
P(y)=\mathbb{E}_{x\sim\hat{P}(x)}P(y|x;\theta)&=\sum_{x\in\mathcal{X}}{P(y|x;\theta)\hat{P}(x)} \\
&=\sum_{x\in\mathcal{X}}\frac{P(y|x;\theta)\hat{P}(x)}{P(x|y)}P(x|y) \\
&=\mathbb{E}_{x\sim P(x|y)}\frac{P(y|x;\theta)\hat{P}(x)}{P(x|y)} \\
&\approx\frac{1}{K}\sum^K_{i=1}{\frac{P(y|x_i;\theta)\hat{P}(x_i)}{P(x_i|y)}}
\end{aligned}$$

$$\mathcal{L}(\theta)\approx-\sum^N_{n=1}{\log{P(y^n|x^n;\theta)}}+\lambda\sum^S_{s=1}{[\log{\hat{P}(y^s)}-\log{\frac{1}{K}\sum^K_{i=1}\frac{\hat{P}(x^s_i)P(y^s|x^s_i\theta)}{P(x^s_i|y^s)}}]^2}$$

## AnoGAN

$$\underset{\theta}{\text{argmin }}\underset{\phi}{\text{argmax }}V(\theta,\phi)=\mathbb{E}_{x\sim p(x)}\bigg[\log{D\big(x;\phi\big)}\bigg]+\mathbb{E}_{z\sim \mathcal{N}(0,1)}\bigg[\log{\Big(1-D\big(G(z;\theta);\phi\big)\Big)}\bigg]$$

$$\begin{aligned}
\mathcal{L}_R(x;\theta)&=\big|x-G_\theta(\tilde{z})\big|_2 \\
\mathcal{L}_D(x;\phi)&=\sum_{i=1}^L{\Big|D_{\phi_{1:i}}(x)-D_{\phi_{1:i}}\big(G(\tilde{z})\big)\Big|_2}
\end{aligned}$$

$$\begin{gathered}
\text{and}\\
\tilde{z}=\underset{z\sim \mathcal{N}(0,1)}{\text{argmin }}{-\log{P(x|z;\theta)}} \\
\text{where }\tilde{z}\text{ is estimated }z\text{ by back-propagation, and }L\text{ is a number of layers in }D.
\end{gathered}$$

$$\begin{gathered}
z_{t+1}\leftarrow z_t-\alpha\nabla_{z}J(z,x), \\
\text{where }J(z,x)=|x-G(z)|_2.
\end{gathered}$$

$$\text{Anomaly Score }A(x)=(1-\lambda)\cdot\mathcal{L}_R(x;\theta)+\lambda\cdot\mathcal{L}_D(x;\phi)$$
