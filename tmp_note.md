# Temporal Note for Markdown Equations

## Add one Smoothing

$$
\begin{aligned}
P(w_i|w_{<i}) \approx \frac{C(w_{<i},w_i)+1}{C(w_{<i})+V}
\end{aligned}
$$

$$
\begin{aligned}
P(w_i|w_{<i}) &\approx \frac{C(w_{<i},w_i)+k}{C(w_{<i})+kV} \\
&\approx \frac{C(w_{<i},w_i)+(m/V)}{C(w_{<i})+m}
\end{aligned}
$$

$$
P(w_i|w_{<i}) \approx \frac{C(w_{<i},w_i)+m P(w_i)}{C(w_{<i})+m}
$$

## Auto-regressive and Teacher Forcing on RNNLM

$$
X=argmax_X P(X)=argmax_Y \prod_{i=1}^{n}{P(x_i|x_{<i})}
$$

$$
or
$$
$$
\begin{aligned}
x_i=&argmax_y{P(x|x_{<i})} \\
&\text{where }x_0=BOS
\end{aligned}
$$

$$
\hat{x}_t=argmax_x{P(x_t|x_{<t};\theta)}\text{ where }X=\{x_1,x_2,\cdots,x_n\}
$$

$$
\begin{aligned}
\mathcal{L}(X)&=-\sum_{t=1}^{n+1}{\log{P(x_t|x_{<t};\theta)}} \\
\theta &\leftarrow \theta-\lambda\frac{1}{N}\sum_{i=1}^{N}{\mathcal{L}(X_i)}
\end{aligned}
$$

## Expectation

$$
\begin{aligned}
\mathbb{E}_{x \sim p}[reward(x)]&=\int{reward(x)p(x)}dx \\
&\approx\frac{1}{K}\sum_{i=1}^{K}{reward(x_i)} \\
&\approx reward(x)
\end{aligned}
$$

## Importance Sampling

$$
\begin{aligned}
\mathbb{E}_{X \sim p}[f(x)]&=\int_{x}{f(x)p(x)}dx \\
&=\int_{x}{\Big( f(x)\frac{p(x)}{q(x)}\Big)q(x)}dx \\
&=\mathbb{E}_{X \sim q}[f(x)\frac{p(x)}{q(x)}],
\end{aligned}
$$

$$
\forall q\text{ (pdf) s.t.} q(x)=0 \implies p(x)=0
$$

$$
w(x)=\frac{p(x)}{q(x)}
$$

$$
\begin{aligned}
\mathbb{E}_{X \sim q}[f(x)\frac{p(x)}{q(x)}]&\approx\frac{1}{K}\sum_{i=1}^{K}{f(x_i)\frac{p(x_i)}{q(x_i)}} \\
&=\frac{1}{K}\sum_{i=1}^{K}{f(x_i)w(x_i)} \\
\text{where }& x_i \sim q
\end{aligned}
$$

## GAN

$$
\min_{G}\max_{D}\mathcal{L}(D,G)=\mathbb{E}_{x\sim p_r(x)}[\log{D(x)}]+\mathbb{E}_{z\sim p_z(z)}[\log{(1-D(G(z)))}]
$$

## Policy Iteration

### Policy Evaluation

$$
\begin{aligned}
v_\pi(s) &\doteq \mathbb{E}_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots|S_t=s] \\
&= \mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s] \\
&= \sum_a{\pi(a|s)\sum_{s',r}{P(s',r|s,a)\Big[r+\gamma v_\pi(s')\Big]}}
\end{aligned}
$$

### Policy Improvement

$$
\begin{aligned}
\pi'(s) &\doteq \underset{a}{\text{argmax }}{q_\pi(s,a)} \\
&= \underset{a}{\text{argmax }}{\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a]} \\
&= \underset{a}{\text{argmax }}{\sum_{s',r}{P(s',r|s,a)\Big[r+\gamma v_\pi(s')\Big]}}
\end{aligned}
$$

## Probability

### Discrete Variable

$$
P(X=x)
$$

$$
\sum_{i=1}^N{P(X=x_i)}=\sum_{i=1}^N{P(x_i)}=1
$$

### Continuous Variable

$$
\forall x \in X,~p(x)\ge0.
$$

$$
\text{We don not require that }p(x)\le1.
$$

$$
\int_{-\infty}^{\infty}{p(x)}~dx=1
$$

### Conditional Probability

$$
P(A|B)=\frac{P(A,B)}{P(B)}
$$

#### Conditional Independence

$$
P(A,B)=P(A)P(B)
$$

$$
P(A|B)=\frac{P(A,B)}{P(B)}=P(A)
$$

### Marginal Probability

$$
P(y)=\sum_{x\in\mathcal{X}}{P(y,x)}=\sum_{x\in\mathcal{X}}{P(y|x)P(x)}
$$

### Bayes Theorem

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

### Monty-hall Problem

$$
\begin{aligned}
P(C=2|A=0,B=1)&=\frac{P(A=0,B=1,C=2)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=2)P(A=0,C=2)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=2)P(A=0)P(C=2)}{P(B=1|A=0)P(A=0)} \\
&=\frac{1 \times \frac{1}{3}}{\frac{1}{2}}=\frac{2}{3},\\
\text{where }P(B=1,A=0)=&\frac{1}{2},~P(C=2)=\frac{1}{3},\text{ and }P(B=1|A=0,C=2)=1.
\end{aligned}
$$

$$
\begin{aligned}
P(C=0|A=0,B=1)&=\frac{P(A=0,B=1,C=0)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=0)P(A=0,C=0)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=0)P(A=0)P(C=0)}{P(B=1|A=0)P(A=0)} \\
&=\frac{\frac{1}{2} \times \frac{1}{3}}{\frac{1}{2}}=\frac{1}{3},\\
\text{where }&P(B=1|A=0,C=0)=\frac{1}{2}
\end{aligned}
$$

## Machine Learning

### MLE

$$
\hat{\theta}=\underset{\theta}{\text{argmax }}P(Y|X;\theta)=\underset{\theta}{\text{argmax }}P(Y|X,\theta)
$$

$$
\hat{\theta}=\underset{\theta}{\text{argmax }}P(X;\theta)=\underset{\theta}{\text{argmax }}P(X|\theta)
$$ 

### MAP

$$
\begin{aligned}
\hat{\theta}&=\underset{\theta}{\text{argmax }}P(\theta|X,Y) \\
&=\underset{\theta}{\text{argmax }}P(X,Y,\theta) \\
&=\underset{\theta}{\text{argmax }}P(Y|X;\theta)P(X,\theta) \\
&=\underset{\theta}{\text{argmax }}P(Y|X;\theta)P(X;\theta)P(\theta)
\end{aligned}
$$

$$
\begin{aligned}
\hat{\theta}&=\underset{\theta}{\text{argmax }}P(\theta|X) \\
&=\underset{\theta}{\text{argmax }}P(X,\theta) \\
&=\underset{\theta}{\text{argmax }}P(X;\theta)P(\theta)
\end{aligned}
$$

### Ensemble

$$
P(Y|X)=\mathbb{E}_{\theta\sim P}[P(Y|X;\theta)]\approx\frac{1}{N}\sum_{i=1}^N{P(Y|X;\theta)}
$$

### KL Divergence

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

## Gradient based Optimizations

### Cross Entropy Loss

An objective function by Cross Entropy is

$$
\begin{aligned}
J(\theta)=H(P,P_\theta)&=-\mathbb{E}_{X\sim P(X)}\Big[\mathbb{E}_{Y\sim P(Y|X)}[\log{P(Y|X;\theta)}]\Big] \\
&=-\sum_{x\in\mathcal{X}}{P(x)\sum_{y\in\mathcal{Y}}{P(y|x)\log{P(y|x;\theta)}}} \\
\end{aligned}
$$

By Monte-Carlo Sampling,

$$
\begin{gathered}
\mathcal{B}=\{x,y\}_{i=1}^N \\ \\
J(\theta)\approx-\frac{1}{N}\sum_{i=1}^N{\frac{1}{K}\sum_{j=1}^K{\log{P(y_j|x_i;\theta)}}}
\end{gathered}
$$

To minimize the objective function,

$$
\begin{gathered}
\hat{\theta}=\underset{\theta}{\text{argmin }}J(\theta) \\ \\
\theta \leftarrow \theta-\lambda\nabla_\theta J(\theta)
\end{gathered}
$$

In addition, another objective function by Mean Square Error (MSE) is

$$
J(\theta)=\mathbb{E}_{X\sim P(X)}\Big[\mathbb{E}_{Y\sim P(Y|X)}[(p(y|x)-q_\theta(y|x))^2]\Big]
$$