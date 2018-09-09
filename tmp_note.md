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

## Machine Learning

### MLE

$$
\hat{\theta}=\underset{\theta}{\text{argmax }}P(Y|X;\theta)=\underset{\theta}{\text{argmax }}P(Y|X,\theta)
$$

$$
\hat{\theta}=\underset{\theta}{\text{argmax }}P(X;\theta)=\underset{\theta}{\text{argmax }}P(X|\theta)
$$ 

#### Example

$$
K\sim\mathcal{B}(n,\theta)
$$

$$
\begin{aligned}
P(K=k)&=
\begin{pmatrix}
   n \\
   k
\end{pmatrix}
\theta^k(1-\theta)^{n-k} \\
&=\frac{n!}{k!(n-k)!}\cdot\theta^k(1-\theta)^{n-k}
\end{aligned}
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
