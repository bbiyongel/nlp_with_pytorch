# Expectation

![](/assets/lm_rolling_dice.png)

기대값(expectation)은 reward와 그 reward를 받을 확률을 곱한 값의 총 합을 통해 얻을 수 있습니다. 즉, reward에 대한 가중평균(weighted average)라고 볼 수 있습니다. 주사위의 경우에는 reward의 경우에는 1부터 6까지 받을 수 있지만, 각 reward에 대한 확률은 $$ {1}/{6} $$로 동일합니다.

$$
\begin{aligned}
expected~reward~from~dice&=\sum^6_{x=1}{P(X=x)\times reward(x)} \\
where P(x)=\frac{1}{6}, \forall x~~&and~~reward(x)=x.
\end{aligned}
$$

따라서 실제 주사위의 기대값은 아래와 같이 3.5가 됩니다.

$$
\frac{1}{6}\times(1+2+3+4+5+6)=3.5
$$

또한, 위의 수식은 아래와 같이 표현 할 수 있습니다.

$$
E_{X \sim P}[reward(x)]=\sum^6_{x=1}{P(X=x)\times reward(x)}=3.5
$$

# Monte Carlo Sampling

_**Monte Carlo Sampling**_은 난수를 이용하여 임의의 함수를 근사하는 방법입니다. 예를 들어 임의의 함수 $$f$$가 있을 때, 사실은 해당 함수가 Gaussian distribution을 따르고 있고, 충분히 많은 수의 random number $$x$$를 생성하여, $$f(x)$$를 구한다면, $$f(x)$$의 분포는 역시 gaussian distribution을 따르고 있을 것 입니다. 이와 같이 임의의 함수에 대해서 Monte Carlo 방식을 통해 해당 함수를 근사할 수 있습니다.

![approximation of pi using Monte Carlo](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)

따라서 Monte Carlo sampling을 사용하면 기대값(expectation) 내의 표현을 밖으로 끄집어 낼 수 있습니다. 즉, 주사위의 reward에 대한 기대값을 아래와 같이 simplify할 수 있습니다.

$$
E_{x \sim P(X)}[reward(x)] \approx \frac{1}{N}\sum^N_{i=1}{reward(x_i)}
$$

주사위 reward의 기대값은 $$ N $$번 sampling한 주사위 값의 평균이라고 할 수 있습니다. N값이 무한대에 가까워질 수록 (커질 수록) 해당 값은 실제 기대값 $$3.5$$에 가까워질 것 입니다. 따라서 우리는 경우에 따라서 N이 1인 경우도 가정 해 볼 수 있습니다. 즉, 아래와 같은 수식이 될 수도 있습니다.

$$
E_{X \sim P}[reward(x)] \approx reward(x)=x
$$

위와 같은 가정을 가지고 수식을 simplify할 수 있게 되면, 이후 gradient를 구한다거나 할 때에 수식이 간단해져 매우 편리합니다.

# Policy Gradient

Policy Gradient는 _**Policy based Reinforcement Learning**_ 방식입니다. 알파고를 개발했던 DeepMind에 의해서 유명해진 Deep Q-Learning은 Value based Reinforcement Learning 방식에 속합니다. 실제 딥러닝을 사용하여 두 방식을 사용 할 때에 가장 큰 차이점은, Value based방식은 인공신경망을 사용하여 어떤 action을 하였을 때에 얻을 수 있는 보상을 예측 하도록 훈련하는 것과 달리, policy based 방식은 인공신경망은 어떤 action을 할지 훈련되고 해당 action에 대한 보상\(reward\)를 back-propagation 할 때에 gradient를 통해서 전달해 주는 것이 가장 큰 차이점 입니다. 따라서 어떤 Deep Q-learning의 경우에는 action을 선택하는 것이 deterministic한 것에 비해서, _**Policy Gradient**_방식은 action을 선택 할 때에 stochastic한 process를 거치게 됩니다. Policy Gradient에 대한 수식은 아래와 같습니다.


$$
\pi_\theta(a|s) = P_\theta(a|s) = P(a|s; \theta)
$$


위의 $$\pi$$는 policy\(정책\)을 의미합니다. 그리고 위와 같이 확률로 표현 될 수 있습니다.


$$
\begin{aligned}
J(\theta) &= E_{\pi_\theta}[r] = v_\theta(s_0) \\
&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\pi_\theta(s, a)\mathcal{R}_{s, a}}
\end{aligned}
$$


즉 우리의 objective function은 initial state에서의 expected cumulative reward를 최대\(maximize\)로 하도록 하는 policy\($$\theta$$\)를 찾는 것 입니다.


$$
\theta_{t+1}=\theta_t+\alpha\triangledown_\theta J(\theta)
$$


Gradient Ascent에 따라서, $$\triangledown_\theta J(\theta)$$를 구하여 $$\theta$$를 업데이트 해야 합니다. 여기서 $$ d(s) $$는 markov chain의 stationary distribution으로써 시작점에 상관없이 전체의 trajecotry에서 $$ s $$에 머무르는 시간의 proportion을 의미합니다.


$$
\begin{aligned}
\triangledown_\theta\pi_\theta(s,a)&=\pi_\theta(s,a)\frac{\triangledown_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)} \\
&=\pi_\theta(s,a)\triangledown_\theta\log{\pi_\theta(s,a)}
\end{aligned}
$$


이때, 위와 같이 로그의 미분의 성질을 이용하여 $$ \triangledown_\theta J(\theta) $$를 구할 수 있습니다.


$$
\begin{aligned}
\triangledown_\theta J(\theta)&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\pi_\theta(s,a)}\triangledown_\theta\log{\pi_\theta(s, a)\mathcal{R}_{s,a}} \\
&= E_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}r]
\end{aligned}
$$


_**Policy Gradient Theorem**_에 따르면, 여기서 해당 time-step에 대한 즉각적인 reward\($$ r $$\) 대신에 episode의 종료까지의 총 reward, 즉 $$ Q $$ function을 사용할 수 있습니다.


$$
\triangledown_\theta J(\theta) = E_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}Q^{\pi_\theta}(s,a)]
$$


여기서 바로 Policy Gradients의 진가가 드러납니다. 우리는 policy network에 대해서 gradient를 구하지만, Q-function에 대해서는 gradient를 구할 필요가 없습니다. 즉, 미분이 불가능한 함수라도 reward function으로 사용할 수 있는 것입니다. 이렇게 어떠한 함수도 reward로 사용할 수 있게 됨에 따라, 기존의 cross entropy와 같은 loss function 대신에 좀 더 task에 부합한 함수\(번역의 경우에는 BLEU\)를 사용하여 network를 훈련시킬 수 있게 되었습니다. 위의 수식에서 expectation을 Monte Carlo sampling으로 대체하면 아래와 같이 parameter update를 수행 할 수 있습니다.


$$
\theta \leftarrow \theta + \alpha Q^{\pi_\theta}(s_t,a_t)\triangledown_\theta\log{\pi_\theta(a_t|s_t)}
$$


위의 수식을 NLP에 적용하여 쉽게 설명 해 보면, Monte Carlo 방식을 통해 sampling 된 sentence에 대해서 gradient를 구하고, 그 gradient에 reward를 곱하여 주는 형태입니다. 만약 샘플링 된 해당 문장이 좋은 \(큰 양수\) reward를 받았다면 learning rate $$ \alpha $$에 추가적인 scaling을 통해서 더 큰 step으로 gradient ascending을 할 수 있을 겁니다. 하지만 negative reward를 받게 된다면, gradient는 반대로 적용이 되도록 값이 곱해지게 될 겁니다. 따라서 해당 샘플링 된 문장이 나오지 않도록 parameter $$ \theta $$가 update 됩니다.

따라서 실제 gradient에 따른 local minima를 찾는 것이 아닌, 아래 그림과 같이 실제 reward-objective function에 따른 최적을 값을 찾게 됩니다. 하지만, 기존의 gradient는 방향과 크기를 나타낼 수 있었던 것에 비해서, policy gradient는 기존의 gradient의 방향에 크기(scalar)값을 곱해줌으로써 방향을 직접 지정해 줄 수는 없습니다. 따라서 실제 objective function에 따른 최적의 방향을 스스로 찾아갈 수는 없습니다. 그러므로 매우 훈련이 어렵고 비효율적인 단점을 갖고 있습니다.

![](/assets/rl_sgd_vs_policy_gradients.png)

Policy Gradient에 대한 자세한 설명은 원 논문인 [\[Sutton at el.1999\]](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), 또는 해당 저자가 쓴 textbook ['Reinforcement Learning: An Introduction'](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf)을 참고하거나, [DeepMind David Silver의 YouTube 강의](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)를 참고하면 좋습니다.

## MLE vs RL(Policy Gradients)

$$
\begin{aligned}
&(X, Y) \in \mathcal{B} \\
X&=\{x_1, x_2, \cdots, x_n\} \\
Y&=\{y_0, y_1, \cdots, y_m\}
\end{aligned}
$$

$$
\begin{aligned}
\hat{Y}=argmax_{Y}P(Y|X)
\end{aligned}
$$

$$
\begin{aligned}
\hat{\theta}&=argmax_{\theta}P(\theta|X, Y) \\
&=argmax_{\theta}P(Y|X; \theta)P(\theta) \\
\end{aligned}
$$

$$
\theta \leftarrow \theta + \gamma \nabla J(\theta)
$$

$$
\begin{aligned}
J(\theta)&=-\sum_{(X, Y) \in \mathcal{B}}{\log{P(Y|X;\theta)}} \\
&=-\sum_{(X, Y) \in \mathcal{B}}{\sum_{i = 0}^{m}{\log{P(y_i|X, y_{i}; \theta)}}}
\end{aligned}
$$

$$
\theta \leftarrow \theta - \gamma \sum_{(X, Y) \in \mathcal{B}}{\sum_{i = 0}^{m}{\nabla_\theta\log{P(y_i|X, y_{i}; \theta)}}}
$$

$$
\theta \leftarrow \theta + \alpha Q^{\pi_\theta}(s_t,a_t)\triangledown_\theta\log{\pi_\theta(a_t|s_t)}
$$