# Dual Supervised Learning

이번에 소개할 방법은 [Dual Supervised Learning (DSL)]() 입니다. 이 논무

$$
P(x)P(y|x;\theta_{x \rightarrow y})=P(y)P(x|y;\theta_{y \rightarrow x})
$$
<br>
$$
\begin{aligned}
objective 1: \min_{\theta_{x \rightarrow y}}{\frac{1}{n}\sum^n_{i=1}{\ell_1(f(x_i;\theta_{x \rightarrow y}), y_i)}}, \\
objective 2: \min_{\theta_{y \rightarrow x}}{\frac{1}{n}\sum^n_{i=1}{\ell_1(g(y_i;\theta_{y \rightarrow x}), x_i)}}, \\
s.t.~P(x)P(y|x;\theta_{x \rightarrow y})=P(y)P(x|y;\theta_{y \rightarrow x}), \forall{x, y}.
\end{aligned}
$$
<br>
$$
\mathcal{L}_{duality}=((\log{\hat{P}(x)} + \log{P(y|x;\theta_{x \rightarrow y})}) - (\log{\hat{P}(y)} + \log{P(x|y;\theta_{y \rightarrow x}}))^2
$$
<br>
$$
\begin{aligned}
\theta_{x \rightarrow y} \leftarrow \theta_{x \rightarrow y}-\gamma\nabla_{\theta_{x \rightarrow y}}\frac{1}{n}\sum^m_{j=1}{[\ell_1(f(x_i;\theta_{x \rightarrow y}), y_i)+\lambda_{x \rightarrow y}\mathcal{L}_{duality}]} \\
\theta_{y \rightarrow x} \leftarrow \theta_{y \rightarrow x}-\gamma\nabla_{\theta_{y \rightarrow x}}\frac{1}{n}\sum^m_{j=1}{[\ell_2(g(y_i;\theta_{y \rightarrow x}), x_i)+\lambda_{y \rightarrow x}\mathcal{L}_{duality}]}
\end{aligned}
$$