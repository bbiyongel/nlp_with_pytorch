# 정책 기반 강화학습

## 폴리시 그래디언트 (Policy Gradients)

폴리시 그래디언트는 정책기반 강화학습(Policy based Reinforcement Learning) 방식에 속합니다. 알파고를 개발했던 딥마인드에 의해서 유명해진 딥-큐 학습법(Deep Q-learning)은 가치기반 강화학습(Value based Reinforcement Learning) 방식에 속합니다. 실제 딥러닝을 사용하여 두 방식을 사용 할 때에 가장 큰 차이점은, 가치기반 학습 방식은 인공신경망을 사용하여 어떤 행동(action)을 선택 하였을 때에 얻을 수 있는 보상을 예측 하도록 훈련하는 것과 달리, 정책 기반 방식은 인공신경망은 어떤 행동을 선택 할지 훈련되고 해당 행동에 대한 보상(reward)을 back-propagation 할 때에 그래디언트(gradient)를 통해서 전달해 주는 것이 가장 큰 차이점 입니다. 따라서 어떤 딥-큐러닝의 경우에는 행동을 선택하는 것이 확률적으로 나오지 않는 것에 비해서, 폴리시 그래디언트 방식은 행동을 선택 할 때에 확률적인(stochastic) 프로세스를 거치게 됩니다. 폴리시 그래디언트에 대한 수식은 아래와 같습니다.

$$\pi_\theta(a|s) = P_\theta(a|s) = P(a|s; \theta)$$

위의 $\pi$ 는 정책(policy)을 의미합니다. 즉, 뉴럴 네트워크 웨이트 파라미터 $\theta$ 는 현재 상태 $s$ 가 주어졌을 때, 어떤 행동 $a$ 을 선택해야하는지 확률 분포를 반환 합니다.

$$\begin{aligned}
J(\theta) &= \mathbb{E}_{\pi_\theta}[r] = v_\theta(s_0) \\
&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\pi_\theta(s, a)\mathcal{R}_{s, a}}
\end{aligned}$$

우리의 목표(objective)는 최초 상태(initial state)에서의 기대누적보상(expected cumulative reward)을 최대(maximize)로 하도록 하는 정책 $\theta$ 를 찾는 것 입니다. 최소화 하여야 하는 손실(loss)와 달리 보상은 최대화 하여야 하므로 기존의 그래디언트 디센트 대신에 그래디언트 어센트(ascent)를 사용하여 최적화를 수행 합니다.

$$\theta_{t+1}=\theta_t+\alpha\triangledown_\theta J(\theta)$$

그래디언트 어센트에 따라, $\triangledown_\theta J(\theta)$ 를 구하여 $\theta$ 를 업데이트 해야 합니다. <comment> 여기서 $d(s)$ 는 마코프 체인의 stationary distribution으로써 시작점에 상관없이 전체의 경로(trajecotry)에서 $s$ 에 머무르는 시간의 proportion을 의미합니다. </comment>

$$\begin{aligned}
\triangledown_\theta\pi_\theta(s,a)&=\pi_\theta(s,a)\frac{\triangledown_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)} \\
&=\pi_\theta(s,a)\triangledown_\theta\log{\pi_\theta(s,a)}
\end{aligned}$$

이때, 위의 로그 미분의 성질을 이용하여 아래와 같이 $\triangledown_\theta J(\theta)$ 를 구할 수 있습니다. 이 수식을 해석하면 매 time-step 별 상황 $s$ 이 주어졌을 때 선택 $a$ 할 로그 확률의 그래디언트와 그에 따른 보상(reward)을 곱한 값의 기대값이 됩니다.

$$\begin{aligned}
\triangledown_\theta J(\theta)&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\triangledown_\theta\pi_\theta(s,a)\mathcal{R}_{s,a}}  \\
&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\pi_\theta(s,a)}\triangledown_\theta\log{\pi_\theta(s, a)\mathcal{R}_{s,a}} \\
&= \mathbb{E}_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}r]
\end{aligned}$$

폴리시 그래디언트 정리(Policy Gradient Theorem)에 따르면, 여기서 해당 time-step에 대한 즉각적인 보상 $r$ 대신에 에피소드의 종료까지의 기대 누적 보상, 즉 큐함수(Q function)을 사용할 수 있습니다.

$$\triangledown_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}Q^{\pi_\theta}(s,a)]$$

여기서 바로 폴리시 그래디언트의 진가가 드러납니다. 우리는 폴리시 그래디언트의 뉴럴 네트워크에 대해서는 미분을 계산해야 하지만, 큐함수에 대해서는 미분을 할 필요가 없습니다. 즉, 미분의 가능 여부를 떠나서 임의의 어떠한 함수라도 보상 함수(reward function)로 사용할 수 있는 것입니다. 이렇게 어떠한 함수도 보상 함수로 사용할 수 있게 됨에 따라, 기존의 크로스 엔트로피나 MSE와 같은 손실 함수를 통해 학습(fitting) 시키는 대신, 좀 더 실제 문제에 부합하는 함수(번역의 경우에는 BLEU)를 사용하여 $\theta$ 를 훈련시킬 수 있게 되었습니다. 위의 수식에서 기대값 수식을 몬테카를로 샘플링으로 대체하면 아래와 같이 뉴럴 네트워크 파라미터 업데이트를 수행 할 수 있습니다.

$$\theta \leftarrow \theta + \alpha Q^{\pi_\theta}(s_t,a_t)\triangledown_\theta\log{\pi_\theta(a_t|s_t)}$$

위의 수식을 좀 더 쉽게 설명 해 보면, 몬테카를로 방식을 통해 샘플링 된 행동(action)들에 대해서 그래디언트를 구하고, 그 그래디언트에 보상을 곱하게 됩니다. 만약 샘플링 된 해당 행동들이 좋은 보상을 받았다면 learning rate $\alpha$ 에 추가적인 곱셈을 통해서 더 큰 스텝으로 그래디언트 어센딩을 수행 할 수 있을 겁니다. 하지만 나쁜 보상값을 받게 된다면, 그래디언트의 반대방향으로 스텝을 갖도록 값이 곱해지게 될 겁니다. 즉, 그래디언트 어센트 대신에 그래디언트 디센트를 수행하는 것과 같은 효과가 날 것 입니다. 따라서 해당 샘플링 된 행동들이 앞으로는 잘 나오지 않도록 뉴럴 네트워크 파라미터 $\theta$ 가 업데이트 됩니다.

따라서 실제 gradient에 따른 local minima(지역최소점)를 찾는 것이 아닌, 아래 그림과 같이 실제 reward-objective function에 따른 최적을 값을 찾게 됩니다. 하지만, 기존의 gradient는 방향과 크기를 나타낼 수 있었던 것에 비해서, policy gradients는 기존의 gradient의 방향에 크기(scalar)값을 곱해줌으로써 방향을 직접 지정해 줄 수는 없습니다. 따라서 실제 목적함수(objective function)에 따른 최적의 방향을 스스로 찾아갈 수는 없습니다. 그러므로 사실 훈련이 어렵고 비효율적인 단점을 갖고 있습니다.

![](../assets/rl_sgd_vs_policy_gradients.png)

Policy Gradient에 대한 자세한 설명은 원 논문인 [\[Sutton at el.1999\]](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), 또는 해당 저자가 쓴 텍스트북 [Reinforcement Learning: An Introduction[Sutton et al.2017]](http://www.incompleteideas.net/book/bookdraft2017nov5.pdf)을 참고하거나, [DeepMind David Silver의 YouTube 강의](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)를 참고하면 좋습니다.

## MLE vs RL(Policy Gradients)

여기서 reward의 역할을 좀 더 직관적으로 설명하고 넘어가도록 하겠습니다.

우리에게 $n$ 개의 시퀀스로 이루어진 입력을 받아, $m$ 개의 시퀀스로 이루어진 출력을 하는 함수를 근사하는 것이 목표로 주어진다고 가정 해 봅니다. 그렇다면 시퀀스 $X$ 와 $Y$ 는 $\mathcal{B}$ 라는 dataset에 존재 합니다.

$$\begin{aligned}
\mathcal{B} &= \{(X_i, Y_i)\}_{i=1}^{N}\\
X&=\{x_1, x_2, \cdots, x_n\} \\
Y&=\{y_0, y_1, \cdots, y_m\}
\end{aligned}$$

근사하여 얻은 함수는 임의의 시퀀스 $X$ 가 주어졌을 때, $\hat{Y}$ 를 반환하도록 잘 학습되어 있을 겁니다.

$$\begin{aligned}
\hat{Y}=argmax_{Y}P(Y|X)
\end{aligned}$$

그럼 해당 함수를 근사하기 위해서 우리는 parameter $\theta$ 를 학습해야 합니다. $\theta$ 는 아래와 같이 Maximum Likelihood Estimation(MLE)를 통해서 얻어질 수 있습니다.

$$\begin{aligned}
\hat{\theta}&=argmax_{\theta}P(\theta|X, Y) \\
&=argmax_{\theta}P(Y|X; \theta)P(\theta) \\
\end{aligned}$$

$\theta$ 에 대해 MLE를 수행하기 위해서 우리는 목적함수(objective function)을 아래와 같이 정의 합니다. 아래는 cross entropy loss를 목적함수로 정의 한 것 입니다. 우리의 목표는 목적함수를 최소화(minimize)하는 것 입니다.

$$\begin{aligned}
J(\theta)&=-\sum_{(X, Y) \in \mathcal{B}}{P(Y|X)\log{P(Y|X;\theta)}} \\
&=-\frac{1}{N}\sum_{(X, Y) \in \mathcal{B}}{\sum_{i = 0}^{m}{\log{P(y_i|X, y_{<i}; \theta)}}}
\end{aligned}$$

위의 수식에서 $P(Y|X)$ 는 훈련 데이터셋에 존재하므로 보통 $1$이라고 할 수 있습니다. 따라서 수식에서 생략 할 수 있습니다. 위에서 정의한 목적함수를 최소화 하여야 하기 때문에, gradient descent를 통해 지역최소점(local minima)를 찾아내어 전역최소점(global minima)에 근사(approximation)할 수 있습니다. 해당 수식은 아래와 같습니다.

$$\begin{aligned}
\theta &\leftarrow \theta - \gamma \nabla J(\theta) \\
\theta &\leftarrow \theta + \gamma \frac{1}{N}\sum_{(X, Y) \in \mathcal{B}}{\sum_{i = 0}^{m}{\nabla_\theta\log{P(y_i|X, y_{i}; \theta)}}}
\end{aligned}$$

우리는 위의 수식에서 learning rate $\gamma$ 를 통해 update step size를 조절 하는 것을 확인할 수 있습니다. 아래는 policy gradients에 기반하여 expected cumulative reward를 최대로 하는 gradient ascent 수식 입니다. Reward를 최대화(maximization)해야 하기 때문에 gradient acsent를 사용하는 것을 볼 수 있습니다.

$$\begin{aligned}
\theta &\leftarrow \theta + \alpha\nabla J(\theta) \\
\theta &\leftarrow \theta + \alpha Q^{\pi_\theta}(s_t,a_t)\triangledown_\theta\log{\pi_\theta(a_t|s_t)}
\end{aligned}$$

위의 수식에서도 이전 MLE의 gradient descent 수식과 마찬가지로, $\alpha$ 와 $Q^{\pi_\theta}(s_t,a_t)$ 가 gradient 앞에 붙어서 learning rate역할을 하는 것을 볼 수 있습니다. 따라서 reward에 따라서 해당 action들로부터 배우는 것을 더욱 강화하거나 반대방향으로 부정할 수 있는 것 입니다. 마치 좀 더 쉽게 비약적으로 설명하면 결과에 따라서 동적으로 learning rate를 알맞게 조절해 주는 것이라고 이해할 수 있습니다.

## REINFORCE with baseline

만약 위의 policy gradient를 수행 할 때, 보상이 항상 양수인 경우는 어떻게 동작할까요? 

예를 들어 우리가 학교에서 100점 만점의 시험을 보았다고 가정해 보겠습니다. 시험 점수는 0점에서부터 100점까지 분포가 되어 평균 점수 근처에 있을 것입니다. 따라서 대부분의 학생들은 양의 보상을 받게 됩니다. 그럼 위의 기존 policy gradient는 항상 양의 보상을 받아 학생에게 박수쳐주며 해당 정책(policy)를 더욱 독려 할 것 입니다. 하지만 알고보면 평균점수 50점 일 때, 시험점수 10점은 매우 나쁜 점수라고 할 수 있습니다. 따라서 받수 받기보단 혼나서, 기존 정책(policy)의 반대방향으로 학습해야 합니다. 하지만 평균점수 50점일 때 시험점수 70점은 여전히 좋은 점수이고 박수 받아 마땅 합니다. 마찬가지로 평균 50점일 때 시험점수 90점은 70점보다 더 훌륭한 점수이고 박수갈채를 받아야 합니다.

주어진 상황에서 받아 마땅한 누적보상이 있기 때문에, 우리는 이를 바탕으로 현재 정책이 얼마나 훌륭한지 평가 할 수 있습니다. 이를 아래와 같이 policy gradient 수식으로 표현할 수 있습니다.

$$\theta \leftarrow \theta + \alpha\Big(G_t-b(S_t)\Big)\nabla_\theta\log{\pi_\theta(a_t|s_t)}$$
