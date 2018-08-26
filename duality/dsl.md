# Dual Supervised Learning

이번에 소개할 방법은 [Dual Supervised Learning \(DSL\) \[Xia et al.2017\]](https://arxiv.org/pdf/1707.00415.pdf) 입니다. 이 방법은 기존의 Teacher Forcing의 문제로 생기는 어려움을 강화학습을 사용하지 않고, Daulity로부터 regularization term을 이끌어내어 해결하였습니다.

베이즈 정리\(Bayes Theorem\)에 따라서 우리는 아래의 수식이 언제나 성립함을 알고 있습니다.


$$
\begin{aligned}
P(Y|X)&=\frac{P(X|Y)P(Y)}{P(X)} \\
P(Y|X)P(X)&=P(X|Y)P(Y)
\end{aligned}
$$


따라서 위의 수식을 따라서, 우리의 데이터셋을 통해 훈련한 모델들은 아래와 같은 수식을 만족해야 합니다.


$$
P(x)P(y|x;\theta_{x \rightarrow y})=P(y)P(x|y;\theta_{y \rightarrow x})
$$


이 전제를 우리의 번역 훈련을 위한 목표에 적용하면 다음과 같습니다.


$$
\begin{aligned}
objective 1: \min_{\theta_{x \rightarrow y}}{\frac{1}{n}\sum^n_{i=1}{\ell_1(f(x_i;\theta_{x \rightarrow y}), y_i)}}, \\
objective 2: \min_{\theta_{y \rightarrow x}}{\frac{1}{n}\sum^n_{i=1}{\ell_1(g(y_i;\theta_{y \rightarrow x}), x_i)}}, \\
s.t.~P(x)P(y|x;\theta_{x \rightarrow y})=P(y)P(x|y;\theta_{y \rightarrow x}), \forall{x, y}.
\end{aligned}
$$


위의 수식을 해석하면, 목표\(objective1\)은 베이즈 정리에 따른 제약조건을 만족함과 동시에, $\ell_1$을 최소화\(minimize\) 하도록 해야 합니다. $\ell_1$은 번역함수 $f$에 입력 $x_i$를 넣어 나온 반환값과 $y_i$ 사이의 손실\(loss\)를 의미 합니다. 마찬가지로, $\ell_2$도 번역함수 $g$에 대해 같은 작업을 수행하고 최소화하여 목표\(objective2\)를 만족해야 합니다.


$$
\mathcal{L}_{duality}=((\log{\hat{P}(x)} + \log{P(y|x;\theta_{x \rightarrow y})}) - (\log{\hat{P}(y)} + \log{P(x|y;\theta_{y \rightarrow x})})^2
$$


그러므로 우리는 $\mathcal{L}_{duality}$와 같이 베이즈 정리에 따른 제약조건의 양 변의 값의 차이를 최소화\(minimize\)하도록 하는 MSE 손실함수\(loss function\)을 만들 수 있습니다. 위의 수식에서 우리가 동시에 훈련시키는 신경망 네트워크 파라미터를 통해 $\log{P(y|x;\theta_{x \rightarrow y})}$와 $\log{P(x|y;\theta_{y \rightarrow x})}$를 구하고, 단방향\(monolingual\) corpus를 통해 별도로 이미 훈련시켜 놓은 언어모델을 통해 $\log{\hat{P}(x)}$와 $\log{\hat{P}(y)}$를 근사\(approximation\)할 수 있습니다.

이 부가적인 제약조건의 손실함수를 기존의 목적함수\(objective function\)에 추가하여 동시에 minimize 하도록 하면, 아래와 같이 표현 할 수 있습니다.


$$
\begin{aligned}
\theta_{x \rightarrow y} \leftarrow \theta_{x \rightarrow y}-\gamma\nabla_{\theta_{x \rightarrow y}}\frac{1}{n}\sum^m_{j=1}{[\ell_1(f(x_i;\theta_{x \rightarrow y}), y_i)+\lambda_{x \rightarrow y}\mathcal{L}_{duality}]} \\
\theta_{y \rightarrow x} \leftarrow \theta_{y \rightarrow x}-\gamma\nabla_{\theta_{y \rightarrow x}}\frac{1}{n}\sum^m_{j=1}{[\ell_2(g(y_i;\theta_{y \rightarrow x}), x_i)+\lambda_{y \rightarrow x}\mathcal{L}_{duality}]}
\end{aligned}
$$


여기서 $\lambda$는 Lagrange multipliers로써, 고정된 값의 hyper-parameter 입니다. 실험 결과 $\lambda=0.01$ 일 때, 가장 좋은 성능을 나타낼 수 있었습니다.

![](./assets/duality-dsl-eval.png)

위의 테이블과 같이 기존의 Teacher Forcing 아래의 cross entropy 방식\(\[1\]번\)과 Minimum Risk Training\(MRT\) 방식\(\[2\]번\) 보다 더 높은 성능을 보입니다.

이 방법은 강화학습과 같이 비효율적이고 훈련이 까다로운 방식을 벗어나서 regularization term을 추가하여 강화학습을 상회하는 성능을 얻어낸 것이 주목할 점이라고 할 수 있습니다.

