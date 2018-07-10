# Expectation

![](/assets/lm_rolling_dice.png)

기대값(expectation)은 보상(reward)과 그 보상을 받을 확률을 곱한 값의 총 합을 통해 얻을 수 있습니다. 즉, reward에 대한 가중합(weighted sum)라고 볼 수 있습니다. 주사위의 경우에는 reward의 경우에는 1부터 6까지 받을 수 있지만, 각 reward에 대한 확률은 $$ {1}/{6} $$로 동일합니다.

$$
\begin{aligned}
expected~reward~from~dice&=\sum^6_{x=1}{P(X=x)\times reward(x)} \\
where~P(x)=\frac{1}{6}, \forall x~~&and~~reward(x)=x.
\end{aligned}
$$

따라서 실제 주사위의 기대값은 아래와 같이 3.5가 됩니다.

$$
\frac{1}{6}\times(1+2+3+4+5+6)=3.5
$$

또한, 위의 수식은 아래와 같이 표현 할 수 있습니다.

$$
\mathbb{E}_{X \sim P}[reward(x)]=\sum^6_{x=1}{P(X=x)\times reward(x)}=3.5
$$

주사위의 경우에는 discrete variable을 다루는 확률 분포이고, continuous variable의 경우에는 적분을 통해 우리는 기대값을 구할 수 있습니다.

# Monte Carlo Sampling

Monte Carlo Sampling은 난수를 이용하여 임의의 함수를 근사하는 방법입니다. 예를 들어 임의의 함수 $$f$$가 있을 때, 사실은 해당 함수가 Gaussian distribution을 따르고 있고, 충분히 많은 수의 random number $$x$$를 생성하여, $$f(x)$$를 구한다면, $$f(x)$$의 분포는 역시 gaussian distribution을 따르고 있을 것 입니다. 이와 같이 임의의 함수에 대해서 Monte Carlo 방식을 통해 해당 함수를 근사할 수 있습니다.

![approximation of pi using Monte Carlo](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)

따라서 Monte Carlo sampling을 사용하면 기대값(expectation) 내의 표현을 밖으로 꺼낼 수 있습니다. 즉, 주사위의 reward에 대한 기대값을 아래와 같이 간단히(simplify) 표현할 수 있습니다.

$$
\mathbb{E}_{X \sim P}[reward(x)] \approx \frac{1}{N}\sum^N_{i=1}{reward(x_i)}
$$

주사위 reward의 기대값은 $$ N $$번 sampling한 주사위 값의 평균이라고 할 수 있습니다. 실제로 $$N$$이 무한대에 가까워질 수록 (커질 수록) 해당 값은 실제 기대값 $$3.5$$에 가까워질 것 입니다. 따라서 우리는 경우에 따라서 $$N=1$$인 경우도 가정 해 볼 수 있습니다. 즉, 아래와 같은 수식이 될 수도 있습니다.

$$
\mathbb{E}_{X \sim P}[reward(x)] \approx reward(x)=x
$$

위와 같은 가정을 가지고 수식을 간단히 표현할 수 있게 되면, 이후 gradient를 구한다거나 할 때에 수식이 간단해져 매우 편리합니다.

# Brief About Reinforcement Learning

강화학습은 매우 방대하고 유서 깊은 학문입니다. 이 챕터에서 모든 내용을 상세하게 다루기엔 무리가 있습니다. 따라서 우리가 다룰 policy gradient를 알기 위해서 필요한 정도로 가볍게 짚고 넘어가고자 합니다. 더 자세한 내용은 Sutton 교수의 강화학습 책 [Reinforcement Learning: An Introduction [Sutton et al.2017]](http://www.incompleteideas.net/book/bookdraft2017nov5.pdf)을 참고하면 좋습니다.

## Universe

먼저, 강화학습은 어떤 형태로 구성이 되어 있는지 이야기 해 보겠습니다. 강화학습은 어떠한 객체가 주어진 환경에서 상황에 따라 어떻게 행동해야 할지 학습하는 방법에 대한 학문입니다. 그러므로 강화학습은 아래와 같은 요소들로 구성되어 동작 합니다.

![](/assets/rl-universe.png)

처음 상황(state) $$S_t$$ $$(t=0)$$을 받아서 agent는 자신의 정책에 따라 행동(action) $$A_t$$를 선택합니다. 그럼 environment는 agent로부터 action $$A_t$$를 받아 보상(reward) $$R_{t+1}$$과 새롭게 바뀐 상황(state) $$S_{t+1}$$을 반환합니다. 그럼 agent는 다시 그것을 받아 action을 선택하게 됩니다. 따라서 아래와 같이 state, action reward가 시퀀스(sequence)로 주어지게 됩니다.

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,S_3,A_3,\cdots
$$

특정 조건이 만족되면 environment는 이 시퀀스를 종료하고, 이를 하나의 에피소드(episode)라고 합니다. 반복되는 에피소드 하에서 agent를 강화학습을 통해 적절한 행동(보상을 최대로)을 하도록 훈련하는 것이 우리의 목표 입니다.

## Markov Decision Process

그리고 여기에 더해서 Markov Decision Process (MDP)라고 하는 성질을 도입합니다. 

우리는 온세상의 현재(present) $$T=t$$ 이 순간을 하나의 상황(state)으로 정의하게 됩니다. 그럼 현재상황(present state)이 주어졌을 때, 미래 $$T>t$$는 과거 $$T<t$$로부터 독립(independent)이라고 가정 합니다. 그럼 이제 우리 세상은 Markov process(마코프 프로세스)상에서 움직인다고 할 수 있습니다. 이제 우리는 현재 상황에서 미래 상황으로 바뀔 확률을 아래와 같이 수식으로 표현할 수 있습니다.

$$
P(S'|S)
$$

여기에 Markov decision process(마코프 결정 프로세스)는 결정을 내리는 과정, 즉 행동을 선택하는 과정이 추가 된 것 입니다. 풀어 설명하면, 현재 상황에서 어떤 행동을 선택 하였을 때 미래 상황으로 바뀔 확률이라고 할 수 있습니다. 따라서 그 수식은 아래와 같이 표현 할 수 있습니다.

$$
P(S'|S, A)
$$

이제 우리는 MDP 아래에서 environment(환경)와 agent(에이전트)가 state와 reward, action을 주고 받으며 나아가는 과정을 표현 할 수 있습니다.

![](/assets/rl-rpc.png)

## Reward

앞서, agent가 어떤 행동을 선택 하였을 때, 환경(environment)으로부터 보상(reward)을 받는다고 하였습니다. 이때 우리는 $$G_t$$를 어떤 시점으로부터 받은 보상의 총 합이라고 정의 합니다. 따라서 $$G_t$$는 아래와 같이 정의 됩니다.

$$
G_t	\doteq R_{t+1}+R_{t+2}+R_{t+3}+\cdots+R_{T}
$$

이때 우리는 discount factor $$\gamma$$를 도입하여 수식을 다르게 표현 할 수도 있습니다. Discount factor가 도입됨에 따라서 우리는 먼 미래의 보상보다 가까운 미래의 보상을 좀 더 중시해서 다룰 수 있게 됩니다.

$$
G_t	\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots=\sum_{k=0}^\infty{\gamma^k R_{t+k+1}}
$$

## Policy

Agent는 주어진 상황(state)에서 앞으로 받을 보상(reward)의 총 합을 최대로 하도록 행동해야 합니다. 마치 우리가 눈 앞의 즐거움을 참고 미래를 위해서 시험공부를 하듯이, 눈 앞의 작은 손해보다 먼 미래까지의 보상의 총 합이 최대가 되는 것이 중요 합니다. 따라서 agent는 어떠한 상황에서 어떻게 행동을 해야겠다라는 기준이 있어야 합니다.

사람도 상황에 따라서 즉흥적으로 임의의 행동을 선택하기보단 자신의 머릿속에 훈련되어 있는대로 행동하기 마련입니다. 무릎에 작은 충격이 왔을 때 다리를 들어올리는 '무릎반사'와 같은 무의식적인 행동에서부터, 파란불이 들어왔을 때 길을 건너는 행동, 어려운 수학 문제가 주어지면 답을 얻는 과정까지, 모두 주어진 상황에 대해서 행동해야 하는 기준이 있습니다.

정책(policy)은 이렇게 agent가 상황에 따라서 어떻게 행동을 해야 할 지 확률적으로 나타낸 기준 입니다. 같은 상황이 주어졌을 때 항상 같은 행동만 반복하는 것이 아니라 확률적으로 행동을 선택한다고 할 수 있습니다. 물론 확률을 $$1$$로 표현하면 같은 행동만 반복하게 될 겁니다.

정책은 상황에 따른 가능한 행동에 대한 확률의 맵핑(mapping) 함수라고 할 수 있습니다. 수식으로는 아래와 같이 표현 합니다.

$$
\pi(a|s)=P(A_t=a|S_t=s)
$$

따라서 우리는 마음속의 정책에 따라 비가 오는 상황(state)에서 자장면과 짬뽕 중에 어떤 음식을 먹을지 확률적으로 선택 할 수 있고, 맑은 날에도 다른 확률 분포 중에서 선택 할 수 있습니다.

![](/assets/rl-policy-choice.png)

## Value Function

가치함수(value function)는 주어진 정책(policy) $$\pi$$ 아래에서 상황(state) $$s$$에서부터 앞으로 얻을 수 있는 보상(reward) 총 합의 기대값을 표현합니다. $$v_\pi(s)$$라고 표기하며, 아래와 같이 수식으로 표현 될 수 있습니다.

$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t|S_t=s]=\mathbb{E}_\pi\bigg[\sum_{k=0}^\infty{\gamma^k R_{t+k+1}\Big|S_t=s}\bigg], \forall s \in \mathcal{S}
$$

앞으로 얻을 수 있는 보상의 총 합의 기대값은 기대누적보상(expected cumulative reward)라고 표현하기도 합니다.

## Action-Value Function (Q-Function)

행동가치함수(action-value function)은 큐함수(Q-function)라고 불리기도 하며, 주어진 정책 $$\pi$$ 아래 상황(state) $$s$$에서 행동(action) $$a$$를 선택 하였을 때 앞으로 얻을 수 있는 보상(reward)의 총 합의 기대값(expected cumulative reward, 기대누적보상)을 표현합니다. 가치함수는 어떤 상황 $$s$$에서 어떤 행동을 선택하는 것에 관계 없이 얻을 수 있는 누적 보상의 기대값이라고 한다면, 행동가치함수는 어떤 행동을 선택하는가에 대한 개념이 추가 된 것 입니다.

상황과 행동에 따른 기대누적보상을 나타내는 행동가치함수의 수식은 아래와 같습니다.

$$
q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t|S_t=s,A_t=a]=\mathbb{E}_\pi\bigg[\sum_{k=0}^\infty{\gamma^k R_{t+k+1}\Big|S_t=s,A_t=a}\bigg]
$$

## Q-learning

우리는 올바른 행동가치함수를 알고 있다면, 어떠한 상황이 닥쳐도 항상 기대누적보상을 최대화(maximize)하는 행복한 선택을 할 수 있을 것 입니다. 따라서 행동가치함수를 잘 학습하는 것을 Q-learning(큐러닝)이라고 합니다.

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha\Big[\overbrace{R_{t+1}+\gamma\max_aQ(S_{t+1},a)}^{\text{Target}}-\overbrace{Q(S_t,A_t)}^{\text{Current}}\Big]
$$

위 수식처럼 target과 current 사이의 차이를 줄이면, 결국 올바른 큐함수를 학습하게 될 것 입니다.

## Deep Q-learning (DQN)

큐함수를 배울 때 상황(state) 공간과 행동(action) 공간이 너무 커서 상황과 행동이 희소한(sparse) 경우에는 문제가 생깁니다. 훈련 과정에서 희소성(sparseness)으로 인해 잘 볼 수 없기 때문 입니다. 따라서 우리는 상황과 행동을 discrete한 별개의 값으로 다루되, 큐함수를 근사(approximation)함으로써 문제를 해결할 수 있습니다.

생각 해 보면, 아까 비가 올 때 자장면과 짬뽕을 선택하는 문제도, 비가 5mm가 오는것과 10mm가 오는 것은 비슷한 상황이며, 100mm 오는 것과는 상대적으로 다른 상황이라고 할 수 있습니다. 하지만 해가 짱짱한 맑은 날에 비해서 비가 5mm 오는 거소가 100mm 오는 것은 비슷한 상황이라고도 할 수 있습니다.

이처럼 상황과 행동을 근사하여 문제를 해결한다고 할 때, 신경망(neural network)은 매우 훌륭한 해결 방법이 될 수 있습니다. 딥마인드(DeepMind)는 신경망을 사용하여 근사한 큐러닝을 통해 아타리(Atari) 게임을 훌륭하게 플레이하는 강화학습 방법을 제시하였고, 이를 deep Q-learning (or DQN)이라고 이름 붙였습니다.

![](/assets/rl-atari.png)

$$
Q(S_t,A_t) \leftarrow \underbrace{Q(S_t,A_t)}_{\text{Approximated}}+\alpha\Big[R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)\Big]
$$

위의 수식처럼 큐함수 부분을 신경망을 통해 근사(approximate)함으로써 희소성(sparseness)문제를 해결하였습니다.

# Policy Gradients

Policy Gradients는 정책기반 강화학습(Policy based Reinforcement Learning) 방식에 속합니다. 알파고를 개발했던 DeepMind에 의해서 유명해진 Deep Q-Learning은 가치기반 강화학습(Value based Reinforcement Learning) 방식에 속합니다. 실제 딥러닝을 사용하여 두 방식을 사용 할 때에 가장 큰 차이점은, Value based방식은 인공신경망을 사용하여 어떤 action을 하였을 때에 얻을 수 있는 보상을 예측 하도록 훈련하는 것과 달리, policy based 방식은 인공신경망은 어떤 action을 할지 훈련되고 해당 action에 대한 보상\(reward\)를 back-propagation 할 때에 gradient를 통해서 전달해 주는 것이 가장 큰 차이점 입니다. 따라서 어떤 Deep Q-learning의 경우에는 action을 선택하는 것이 deterministic한 것에 비해서, Policy Gradient 방식은 action을 선택 할 때에 stochastic한 process를 거치게 됩니다. Policy Gradient에 대한 수식은 아래와 같습니다.


$$
\pi_\theta(a|s) = P_\theta(a|s) = P(a|s; \theta)
$$


위의 $$\pi$$는 정책(policy)을 의미합니다. 즉, 신경망 $$\theta$$는 현재 상황(state) $$s$$가 주어졌을 때, 어떤 선택(action) $$a$$를 해야할 지 확률을 반환 합니다.


$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\pi_\theta}[r] = v_\theta(s_0) \\
&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\pi_\theta(s, a)\mathcal{R}_{s, a}}
\end{aligned}
$$


우리의 목표(objective)는 최초 상황(initial state)에서의 기대누적보상(expected cumulative reward)을 최대\(maximize\)로 하도록 하는 policy\($$\theta$$\)를 찾는 것 입니다. 최소화 하여야 하는 손실(loss)와 달리 보상(reward)는 최대화 하여야 하므로 기존의 gradient descent 대신에 gradient ascent를 사용하여 최적화(optimization)을 수행 합니다.

$$
\theta_{t+1}=\theta_t+\alpha\triangledown_\theta J(\theta)
$$

Gradient ascent에 따라서, $$\triangledown_\theta J(\theta)$$를 구하여 $$\theta$$를 업데이트 해야 합니다. 여기서 $$ d(s) $$는 markov chain의 stationary distribution으로써 시작점에 상관없이 전체의 trajecotry에서 $$ s $$에 머무르는 시간의 proportion을 의미합니다.


$$
\begin{aligned}
\triangledown_\theta\pi_\theta(s,a)&=\pi_\theta(s,a)\frac{\triangledown_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)} \\
&=\pi_\theta(s,a)\triangledown_\theta\log{\pi_\theta(s,a)}
\end{aligned}
$$


이때, 위의 로그 미분의 성질을 이용하여 아래와 같이 $$ \triangledown_\theta J(\theta) $$를 구할 수 있습니다. 이 수식을 해석하면 매 time-step 별 상황($$s$$)이 주어졌을 때 선택($$a$$)할 로그 확률의 gradient에, 그에 따른 보상(reward)을 곱한 값의 기대값이 됩니다.


$$
\begin{aligned}
\triangledown_\theta J(\theta)&=\sum_{s \in \mathcal{S}}{d(s)}\sum_{a \in \mathcal{A}}{\pi_\theta(s,a)}\triangledown_\theta\log{\pi_\theta(s, a)\mathcal{R}_{s,a}} \\
&= \mathbb{E}_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}r]
\end{aligned}
$$


_**Policy Gradient Theorem**_에 따르면, 여기서 해당 time-step에 대한 즉각적인 reward\($$ r $$\) 대신에 episode의 종료까지의 총 reward, 즉 $$ Q $$ function을 사용할 수 있습니다.


$$
\triangledown_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}Q^{\pi_\theta}(s,a)]
$$


여기서 바로 Policy Gradients의 진가가 드러납니다. 우리는 policy network에 대해서 gradient를 구하지만, Q-function에 대해서는 gradient를 구할 필요가 없습니다. 즉, 미분의 가능 여부를 떠나서 임의의 어떠한 함수라도 보상 함수(reward function)로 사용할 수 있는 것입니다. 이렇게 어떠한 함수도 reward로 사용할 수 있게 됨에 따라, 기존의 단순히 cross entropy와 같은 손실 함수(loss function)에 학습(fitting) 시키는 대신에 좀 더 실제 문제에 부합하는 함수\(번역의 경우에는 BLEU\)를 사용하여 $$\theta$$를 훈련시킬 수 있게 되었습니다. 위의 수식에서 기대값 수식을 Monte Carlo sampling으로 대체하면 아래와 같이 parameter update를 수행 할 수 있습니다.

$$
\theta \leftarrow \theta + \alpha Q^{\pi_\theta}(s_t,a_t)\triangledown_\theta\log{\pi_\theta(a_t|s_t)}
$$


위의 수식을 좀 더 쉽게 설명 해 보면, Monte Carlo 방식을 통해 sampling 된 action들에 대해서 gradient를 구하고, 그 gradient에 reward를 곱하여 주는 형태입니다. 만약 샘플링 된 해당 action들이 좋은 \(큰 양수\) reward를 받았다면 learning rate $$ \alpha $$에 추가적인 곱셈을 통해서 더 큰 step으로 gradient ascending을 할 수 있을 겁니다. 하지만 negative reward를 받게 된다면, gradient의 반대방향으로 step을 갖도록 값이 곱해지게 될 겁니다. 따라서 해당 샘플링 된 action들이 앞으로는 잘 나오지 않도록 parameter $$ \theta $$가 update 됩니다.

따라서 실제 gradient에 따른 local minima(지역최소점)를 찾는 것이 아닌, 아래 그림과 같이 실제 reward-objective function에 따른 최적을 값을 찾게 됩니다. 하지만, 기존의 gradient는 방향과 크기를 나타낼 수 있었던 것에 비해서, policy gradients는 기존의 gradient의 방향에 크기(scalar)값을 곱해줌으로써 방향을 직접 지정해 줄 수는 없습니다. 따라서 실제 목적함수(objective function)에 따른 최적의 방향을 스스로 찾아갈 수는 없습니다. 그러므로 사실 훈련이 어렵고 비효율적인 단점을 갖고 있습니다.

![](/assets/rl_sgd_vs_policy_gradients.png)

Policy Gradient에 대한 자세한 설명은 원 논문인 [\[Sutton at el.1999\]](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), 또는 해당 저자가 쓴 텍스트북 [Reinforcement Learning: An Introduction[Sutton et al.2017]](http://www.incompleteideas.net/book/bookdraft2017nov5.pdf)을 참고하거나, [DeepMind David Silver의 YouTube 강의](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)를 참고하면 좋습니다.

## MLE vs RL(Policy Gradients)

여기서 reward의 역할을 좀 더 직관적으로 설명하고 넘어가도록 하겠습니다.

우리에게 $$n$$개의 시퀀스로 이루어진 입력을 받아, $$m$$개의 시퀀스로 이루어진 출력을 하는 함수를 근사하는 것이 목표로 주어진다고 가정 해 봅니다. 그렇다면 시퀀스 $$X$$와 $$Y$$는 $$\mathcal{B}$$라는 dataset에 존재 합니다.

$$
\begin{aligned}
\mathcal{B} &= \{(X_i, Y_i)\}_{i=1}^{N}\\
X&=\{x_1, x_2, \cdots, x_n\} \\
Y&=\{y_0, y_1, \cdots, y_m\}
\end{aligned}
$$

근사하여 얻은 함수는 임의의 시퀀스 $$X$$가 주어졌을 때, $$\hat{Y}$$를 반환하도록 잘 학습되어 있을 겁니다.

$$
\begin{aligned}
\hat{Y}=argmax_{Y}P(Y|X)
\end{aligned}
$$

그럼 해당 함수를 근사하기 위해서 우리는 parameter $$\theta$$를 학습해야 합니다. $$\theta$$는 아래와 같이 Maximum Likelihood Estimation(MLE)를 통해서 얻어질 수 있습니다.

$$
\begin{aligned}
\hat{\theta}&=argmax_{\theta}P(\theta|X, Y) \\
&=argmax_{\theta}P(Y|X; \theta)P(\theta) \\
\end{aligned}
$$

$$\theta$$에 대해 MLE를 수행하기 위해서 우리는 목적함수(objective function)을 아래와 같이 정의 합니다. 아래는 cross entropy loss를 목적함수로 정의 한 것 입니다. 우리의 목표는 목적함수를 최소화(minimize)하는 것 입니다.

$$
\begin{aligned}
J(\theta)&=-\frac{1}{N}\sum_{(X, Y) \in \mathcal{B}}{P(Y|X)\log{P(Y|X;\theta)}} \\
&=-\frac{1}{N}\sum_{(X, Y) \in \mathcal{B}}{\sum_{i = 0}^{m}{\log{P(y_i|X, y_{<i}; \theta)}}}
\end{aligned}
$$

위의 수식에서 $$P(Y|X)$$는 훈련 데이터셋에 존재하므로 보통 $$1$$이라고 할 수 있습니다. 따라서 수식에서 생략 할 수 있습니다. 위에서 정의한 목적함수를 최소화 하여야 하기 때문에, gradient descent를 통해 지역최소점(local minima)를 찾아내어 전역최소점(global minima)에 근사(approximation)할 수 있습니다. 해당 수식은 아래와 같습니다.

$$
\begin{aligned}
\theta &\leftarrow \theta - \gamma \nabla J(\theta) \\
\theta &\leftarrow \theta + \gamma \frac{1}{N}\sum_{(X, Y) \in \mathcal{B}}{\sum_{i = 0}^{m}{\nabla_\theta\log{P(y_i|X, y_{i}; \theta)}}}
\end{aligned}
$$

우리는 위의 수식에서 learning rate $$\gamma$$를 통해 update step size를 조절 하는 것을 확인할 수 있습니다. 아래는 policy gradients에 기반하여 expected cumulative reward를 최대로 하는 gradient ascent 수식 입니다. Reward를 최대화(maximization)해야 하기 때문에 gradient acsent를 사용하는 것을 볼 수 있습니다.

$$
\begin{aligned}
\theta &\leftarrow \theta + \alpha\nabla J(\theta) \\
\theta &\leftarrow \theta + \alpha Q^{\pi_\theta}(s_t,a_t)\triangledown_\theta\log{\pi_\theta(a_t|s_t)}
\end{aligned}
$$

위의 수식에서도 이전 MLE의 gradient descent 수식과 마찬가지로, $$\alpha$$와 $$ Q^{\pi_\theta}(s_t,a_t) $$가 gradient 앞에 붙어서 learning rate역할을 하는 것을 볼 수 있습니다. 따라서 reward에 따라서 해당 action들로부터 배우는 것을 더욱 강화하거나 반대방향으로 부정할 수 있는 것 입니다. 마치 좀 더 쉽게 비약적으로 설명하면 결과에 따라서 동적으로 learning rate를 알맞게 조절해 주는 것이라고 이해할 수 있습니다.

## REINFORCE with baseline

만약 위의 policy gradient를 수행 할 때, 보상이 항상 양수인 경우는 어떻게 동작할까요? 

예를 들어 우리가 학교에서 100점 만점의 시험을 보았다고 가정해 보겠습니다. 시험 점수는 0점에서부터 100점까지 분포가 되어 평균 점수 근처에 있을 것입니다. 따라서 대부분의 학생들은 양의 보상을 받게 됩니다. 그럼 위의 기존 policy gradient는 항상 양의 보상을 받아 학생에게 박수쳐주며 해당 정책(policy)를 더욱 독려 할 것 입니다. 하지만 알고보면 평균점수 50점 일 때, 시험점수 10점은 매우 나쁜 점수라고 할 수 있습니다. 따라서 받수 받기보단 혼나서, 기존 정책(policy)의 반대방향으로 학습해야 합니다. 하지만 평균점수 50점일 때 시험점수 70점은 여전히 좋은 점수이고 박수 받아 마땅 합니다. 마찬가지로 평균 50점일 때 시험점수 90점은 70점보다 더 훌륭한 점수이고 박수갈채를 받아야 합니다.

주어진 상황에서 받아 마땅한 누적보상이 있기 때문에, 우리는 이를 바탕으로 현재 정책이 얼마나 훌륭한지 평가 할 수 있습니다. 이를 아래와 같이 policy gradient 수식으로 표현할 수 있습니다.

$$
\theta \leftarrow \theta + \alpha\Big(G_t-b(S_t)\Big)\nabla_\theta\log{\pi_\theta(a_t|s_t)}
$$
