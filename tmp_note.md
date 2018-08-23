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