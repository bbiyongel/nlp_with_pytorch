# Maximum Likelihood Estimation (최대 가능도 추정)

머신러닝의 목표는 보지 못한 데이터(unseen data)에 대한 좋은 예측(prediction)을 하는 것 입니다. 이것을 우리는 generalization(일반화)라고 합니다. 좋은 generalization 성능을 얻기 위해서 우리는 알고자 하는 ground-truth 확률 분포로부터 데이터를 수집(샘플링)하여, 수집된 데이터를 가장 잘 설명하는 확률 분포 모델을 추정함으로써, 알고자 하는 ground-truth 확률 분포를 근사(approximate) 합니다.

![어느 평범한 압정](../assets/basic_math-push_pin.png)

어느날 당신에게 얄미운 친구녀석이 내기를 제안합니다. 압정을 던졌을 때, 압정의 납작한 면이 바닥을 향해 떨어지면 50원을 받고, 반대의 경우에는 20원을 내야하는 게임 입니다. 우리는 이 게임을 참여할지 말아야 할지 결정해야 합니다. 그래서 게임에서 사용 될 압정과 똑같은 압정을 하나 구해서 집에서 100번 던져 봅니다. 그랬더니 납작한 면이 바닥으로 30번 떨어졌습니다. 그럼 우리는 추측해 볼 수 있죠.

$$\begin{aligned}
\mathbb{E}_{x\sim P}[\text{reward(x)}]&=P(\text{x}=\text{flat})\times50+P(\text{x}=\text{sharp})\times(-20) \\
&\approx\frac{30}{100}\times50-\big(1-\frac{30}{100}\big)\times20=15-14=1
\end{aligned}$$

이 게임의 기대값은 1원입니다! 한번 할 때마다 평균적으로 1원을 벌 수 있는 셈 입니다. 얄미운 친구를 골려줄 좋은 기회입니다. 당장 게임을 시작해야 합니다.

그런데 사실은 이미 여러분은 머릿속으로 Maximum Likelihood Estimation을 수행 한 셈 입니다. 압정을 던지는 사건은 0과 1로 결과가 나오는 베르누이(Bernoulli) 분포를 따르게 됩니다. 이를 여러번 반복시행 하게 되면 binomial 분포를 따르게 됩니다. Binomial 분포는 아래와 같이 정의 할 수 있습니다.

$$K\sim\mathcal{B}(n,\theta)$$

Binomial 분포의 파라미터 $\theta$ 가 있을 때, $n$ 번 압정을 던져 $k$ 번 납작한 면이 바닥으로 떨어질 확률은 아래와 같이 구할 수 있습니다.

$$\begin{aligned}
P(K=k)&=
\begin{pmatrix}
   n \\
   k
\end{pmatrix}
\theta^k(1-\theta)^{n-k} \\
&=\frac{n!}{k!(n-k)!}\cdot\theta^k(1-\theta)^{n-k}
\end{aligned}$$

우리는 아까 압정을 실제로 던져 얻은 데이터를 통해 $n=100$ 이고, $k=30$ 임을 알 수 있었습니다. 그럼 이 압정을 던졌을 때 평평한 면이 바닥으로 떨어질 확률을 가장 잘 설명하는 binomial 확률 분포를 위한 파라미터 $\theta$ 에 대한 함수는 아래와 같이 구할 수 있습니다.

$$J(\theta)=\frac{100!}{30!(100-30)!}\cdot\theta^{30}(1-\theta)^{100-30}$$

이때 이 수식의 값을 likelihood(라이클리후드, 가능도, 우도)라고 부릅니다. 여기서는 binomial 분포의 확률 값과 같습니다.

$$J(\theta)=P(n=100,k=30|\theta)$$

이 함수를 실제로 $\theta$ 에 대해서 그래프로 나타내면 아래와 같습니다. $0.3$ 에서 $J(\theta)$ 가 최대가 되는 것을 볼 수 있습니다. 즉, $\theta=0.3$ 일 때, bionomial 분포는 우리가 수집한 데이터 또는 현상을 가장 잘 설명(재현)합니다.

![Likelihood 함수 곡선](../assets/basic_math-binomial.png)

앞서 $J(\theta)$ 는 likelihood라고 하였습니다. 따라서 우리는 likelihood를 최대화(maximize)하도록 $\theta$ 를 추정하기 때문에 maximum likelihood estimation이라고 부르는 것 입니다. 이처럼 likelihood는 주어진 데이터 $\text{x}$ 를 설명하기위한 확률분포 파라미터( $\theta$ )에 대한 함수이고, 아래와 같이 표현 합니다.

$$P(\text{x};\theta)$$

> 여기서 세미콜론은 수학적으로는 조건부 표기와 비슷합니다. 따라서 약간의 관점 차이가 존재할 뿐, $P(\text{x};\theta)=P(\text{x}|\theta)$ 라고 볼 수 있습니다. 또한 $P(\text{y}|\text{x};\theta)$ 의 경우에는 $P(\text{y}|\text{x},\theta)$ 와 같습니다.

따라서 discrete 랜덤 변수 확률 분포에서는 확률값 자체가 likelihood로 표현 될 수 있으며, continuous 랜덤 변수 확률 분포의 경우에는 확률 밀도(probability density)값이 likelihood를 대신합니다. 서로 독립인 $n$ 번 시행을 거쳐 얻은 데이터( $x_1, x_2, \cdots, x_n$ )에 대한 likelihood는 아래와 같이 표현 할 수 있습니다.

$$P(x_1,x_2,\cdots,x_n|\theta)=P(x_1;\theta)P(x_2;\theta)\cdots P(x_n;\theta)=\prod_{i=1}^n{P(x_i;\theta)}$$

이때 우리는 로그(log)를 취하여 곱을 합으로 표현할 수 있습니다.

$$\log{P(x_1,x_2,\cdots,x_n|\theta)}=\sum_{i=1}^n{\log{P(x_i;\theta)}}$$

이처럼 로그를 취하게 되면, 추후 소수점이 너무 작게 표현되어 언더플로우(underflow) 현상이 나는 것을 방지할 수 있을 뿐더러, 덧셈 연산은 곱셈 연산보다 빠르므로 여러가지 이점이 있습니다. 가장 큰 이점은 아래와 같이 가우시안(gaussian) 분포에서의 exponent를 제거할 수 있다는 것 입니다.

$$\begin{gathered}
J(\theta)=\log{\mathcal{N}(x|\mu,\sigma^2)}=-\frac{1}{2}\log{2\pi\sigma^2}-\frac{(x-\mu)^2}{2\sigma^2} \\
\text{where }\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{1/2}}\exp{\Big\{-\frac{1}{2\sigma^2}(x-\mu)^2\Big\}}.
\end{gathered}$$

위와 같이 우리는 likelihood에 로그를 취하여 log-likelihood를 최대화 하도록 합니다. 만약 여기에 $-1$ 을 곱하면 최소화 문제로 치환할 수 있습니다.

## 뉴럴네트워크: 확률분포함수

사실은 지금까지 likelihood 이야기를 한 이유가 바로 뉴럴 네트워크 또한 확률분포함수이기 때문입니다. 기존에 MNIST 분류 문제를 훈련한 classifier 네트워크의 경우에는 0부터 9까지의 레이블(클래스)를 리턴하게 되므로 discrete 랜덤 변수를 다루는 multinoulli 확률 분포가 됩니다. 따라서 마지막 softmax 레이어의 결과값은 각 클래스별 확률 $\hat{y}$ 을 반환 합니다. 또한 실제 정답 레이블 $y$ 의 경우에는 discrete 값이기 때문에 one-hot 벡터가 되는 것 입니다. 우리는 그러므로 뉴럴 네트워크 웨이트 파라미터 $\theta$ 를 통해 훈련 데이터 $X$ 를 잘 설명하도록 그래디언트 디센트(gradient descent, 경사하강법)를 통해 maximum likelihood estimation을 수행 하는 것 입니다.

$$\begin{gathered}
\theta \leftarrow \theta-\lambda\nabla_\theta J(\theta) \\
\text{where }\theta\text{ is network weight parameter, and }J(\theta)\text{ is negative log-likelihood.}
\end{gathered}$$
