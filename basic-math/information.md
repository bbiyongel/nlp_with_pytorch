# 정보 이론(Information Theory)

## 정보량

우리는 흔히 정보량(information)이라는 표현을 사용 합니다. 이때 정보량은 불확실성 또는 놀람의 정도를 나타냅니다. 일어날 확률이 낮을수록 불확실성이 높고 일어났을 때 놀람이 커질 것이기 때문에, 정보량이 높다는 것은 확률이 낮은것을 의미 합니다. 예를 들어 누구나 뻔히 예측(또는 재현)할 수 있는 것은 작은 정보만으로도 표현이 가능할 것입니다.

|정보량|정보|
|-|-|
|매우 낮음|내일 아침에는 해가 동쪽에서 뜹니다.|
|매우 높음|내일 아침에는 해가 서쪽에서 뜹니다.|
|매우 낮음|올 여름 평균 기온은 섭씨 28도로 예상 됩니다.|
|매우 높음|올 여름 평균 기온은 영하 10도로 예상 됩니다.|

위와 같이 일어날 확률이 낮은 일에 대한 문장일수록 많은 정보를 갖고 있음을 알 수 있습니다. 만약 확률이 낮은 사건에 대해 서술한 문장이 맞을수록 우리는 그 정보가 굉장히 소중할 겁니다. 우리는 정보량을 아래와 같이 수식으로 표현할 수 있습니다.

$$I(\text{x})=-\log{P(\text{x})}$$

## 엔트로피

여기서 우리는 정보량의 평균(기대값)을 취할 수 있습니다. 이를 엔트로피(Entropy)라고 부릅니다.

$$H(P)=-\mathbb{E}_{\text{x}\sim P(\text{x})}[\log{P(\text{x})}]=-\sum_{x\in\mathcal{X}}{P(x)\log{P(x)}}$$

이때, 엔트로피는 분포가 얼마나 퍼져(flat)있는지, 샤프(sharp)한지를 가늠해 볼 수 있는 척도라고 볼 수 있습니다. 보통 분산이 작을수록 작을수록 샤프한 모양을 갖게 됩니다. 이 말은 샤프한 확률 분포일수록 특정 값 $x$ 에 대해서 확률이 높다는 것 입니다. 

![flat한 분포와 sharp한 분포](../assets/image_needed.jpeg)

여기서 다시한번 하나의 개념이 추가 됩니다. 아마 기존의 딥러닝을 통해 분류 문제(classification problem)를 해결해보고자 한 독자분들은 익히 들어보셨을 이름입니다. 크로스 엔트로피(cross entropy)는 $P$ 분포함수에서 샘플링한 $x$ 를 통해 $Q$ 분포함수의 평균 정보량을 나타낸 것 입니다. 크로스라는 단어에서 알 수 있듯이, 다른 분포 $P$ 를 사용하여 대상 분포 $Q$ 의 엔트로피를 측정 합니다.

![크로스 엔트로피의 직관적인 표현](../assets/image_needed.jpeg)

$$H(P, Q)=-\mathbb{E}_{\text{x}\sim P(\text{x})}[\log{Q(\text{x})}]=-\sum_{x\in\mathcal{X}}{P(x)\log{Q(x)}}$$

 $-\log$ 를 취하였기 때문에, 분포 $P$ 와 분포 $Q$ 가 비슷한 모양일수록 크로스 엔트로피 $H(P,Q)$ 는 더 작은 값을 갖게 됩니다. 그러므로 우리는 그동안 분류 문제에서 크로스 엔트로피 손실 함수(loss function)를 사용하여, 손실 함수의 값이 최소가 되도록 그래디언트 디센트(gradient descent)를 통해 뉴럴 네트워크를 훈련하여 온 것 입니다. 즉, 우리가 알아내고 싶은 ground-truth 확률 분포 $P$ 에서 샘플링한 데이터 $\text{x}$ 를 통해 뉴럴 네트워크 확률 분포 $P_\theta$ 에 넣어 크로스 엔트로피가 최소가 되도록 그래디언트 디센트를 수행해 온 것 입니다.

$$\begin{gathered}
\theta\leftarrow\theta-\nabla_\theta\mathcal{L}(\theta) \\
\text{where }\mathcal{L}(\theta)=H(P,P_\theta)
\end{gathered}$$

다만, 이것은 값을 통해 바로 확률을 구할 수 있는 discrete 랜덤 변수를 다루는 확률 분포에만 해당되는 이야기 입니다. 보통 continuous 확률 분포의 경우에는 mean square error (MSE) 손실 함수를 통해 훈련 합니다. Continuous 확률 분포의 경우에 크로스 엔트로피를 적용하는 것이 틀린 내용은 아니지만, $x$ 값을 가지고 확률 값을 구할 수 없기 때문에, 우리는 이것이 가능한 discrete 확률 분포에만 크로스 엔트로피를 수행 합니다.

$$\begin{aligned}
KL(P||Q)&=-\mathbb{E}_{\text{x}\sim P(\text{x})}\Big[\log{\frac{Q(\text{x})}{P(\text{x})}}\Big] \\
&=-\sum_{x\in\mathcal{X}}{P(x)\log{\frac{Q(x)}{P(x)}}} \\
&=-\sum_{x\in\mathcal{X}}P(x)\log{Q(x)}-\sum_{x\in\mathcal{X}}{P(x)\log{P(x)}} \\
&=H(P, Q)-H(P)
\end{aligned}$$

위의 수식은 KL-Divergence(Kullback–Leibler divergence, KLD)를 나타낸 것 입니다. KL-Divergence는 두 분포 사이의 괴리를 보여줍니다. 분포 $P$ 와 분포 $Q$ 의 위치에 따라서 KL-Divergence의 값은 달라질 수 있기 때문에, 대칭의 개념이 아니라서 '거리'라고 표현하지는 않습니다. 따라서 두 분포 사이의 차이를 줄이고자 할 때, KL-Divergence를 최소화(minimize)하도록 하는 것은 매우 좋은 전략이 될 것 입니다. 우리는 그럼 크로스 엔트로피 대신에 뉴럴 네트워크를 훈련하고자 할 때, KL-Divergence를 사용하면 훨씬 더 좋지 않을까요?

$$\begin{gathered}
\begin{aligned}
\mathcal{L}(\theta)&=KL(P||P_\theta) \\
&=H(P,P_\theta)-H(P)
\end{aligned} \\
\\
\begin{aligned}
\nabla_\theta\mathcal{L}(\theta)&=\nabla_\theta KL(P||P_\theta) \\
&=\nabla_\theta{H(P,P_\theta)}-\nabla_\theta{H(P)} \\
&=\nabla_\theta H(P, P_\theta)
\end{aligned}
\end{gathered}$$

아쉽게도 KL-Divergence에 뉴럴 네트워크 파라미터 $\theta$ 로 미분하여 보면 크로스 엔트로피(cross entropy)와 같은 미분 결과값이 나오는 것을 알 수 있습니다. 따라서 우리는 크로스 엔트로피를 손실함수(loss function)로 활용하여 그래디언트 디센트를 수행하는 것과 KL-Divergence를 손실함수로 활용하여 그래디언트 디센트를 수행하여 뉴럴네트워크를 훈련시키는 과정이 같다라고 말할 수 있습니다.

## 크로스 엔트로피 vs 로그 라이클리후드 (log likelihood)

이전 섹션에서 우리는 maximum likelihood estimation (MLE)를 통해 뉴럴 네트워크를 훈련한다고 하였습니다. 그런데 지금은 크로스 엔트로피를 손실함수(loss function)로 사용하여 뉴럴 네트워크를 훈련한다고 합니다. 무슨 말일까요? 사실은 같은 말 입니다.

$$\begin{gathered}
(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\sim P(\text{y|x}) \\
\mathcal{B}=\{(x_i,y_i)\}_{i=1}^n \\
\mathcal{L}(\theta)=H(P,P_\theta)=-\frac{1}{n}\sum_{i=1}^n{\sum_{\text{y}\in\mathcal{Y}}{P(\text{y}|x_i)\log{P(\hat{\text{y}}|x_i;\theta)}}} \\
\text{where }P(\text{y}|x_i)=y_i\text{ and }P(\hat{\text{y}}|x_i;\theta)=\hat{y}_i=f_\theta(x_i)\text{.} \\
\end{gathered}$$

위의 수식에서처럼 ground truth 확률 분포 $P$ 로부터 샘플링한 $n$ 개의 데이터 $(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)$ 쌍이 있다고 하겠습니다. 그럼 크로스 엔트로피를 손실 함수로 활용하여, 손실 함수를 최소화 하도록 뉴럴 네트워크를 훈련하게 될 것 입니다. 참고로 이때 $P(\text{y}|x_i), P(\hat{\text{y}}|x_i;\theta)$ 와 같은 꼴은 확률 값을 반환하는 함수가 아닌 분포를 반환하는 함수 입니다.

즉, $\text{y}$ 가 discrete 랜덤 변수라면 one-hot 벡터가 될 것 입니다. 따라서 아래와 같이 예를 들어 4차원의 multinoulli 확률 분포였다면, $\text{y}$ 로부터 샘플링한 샘플들은 4차원의 one-hot 벡터일 것 입니다.

$$\begin{gathered}
\text{If y is discrete random variable, }y_i\text{ would be one-hot vector.} \\
\text{For example, }y_1=[0, 0, 1, 0], \forall y\in\mathbb{R}^{d}\text{, and }d=4.
\end{gathered}$$

그리고 뉴럴 네트워크는 마지막 레이어를 softmax 레이어를 가짐으로써, 샘플당 각 클래스에 대한 확률값을 반환하는 함수 형태가 될 것 입니다. <comment> $P(\hat{\text{y}}|x_i;\theta)$ </comment>

$$\begin{gathered}
\text{If we get }\hat{y}_1=[.2,.5,.1,.2], \\
\mathcal{L}(\theta)=-\frac{1}{n}\Big(y_1\odot\hat{y}_1+y_2\odot\hat{y}_2+\cdots+y_n\odot\hat{y}_n\Big) \\
\\
y_1\odot\hat{y}_1=[0,0,1,0]\times[.2,.5,.1,.2]=0\times.2+0\times.5+1\times.1+0\times.2=.1 \\
\text{where }\odot\text{ means sum of element-wise product.}
\end{gathered}$$

그럼 우리는 위의 수식대로 손실 함수를 계산하게 될 겁니다. 즉, discrete 확률 분포를 계산할 때, 크로스 엔트로피는 log-likelihood의 합에 $-1$ 을 곱한 것과 같습니다. 이를 negative log-likelihood라고 합니다.

$$\begin{gathered}
\begin{aligned}
\mathcal{L}(\theta)=H(P,P_\theta)&=-\frac{1}{n}\sum_{i=1}^n{\sum_{\text{y}\in\mathcal{Y}}{P(\text{y}|x_i)\log{P(\hat{\text{y}}|x_i;\theta)}}} \\
&=-\frac{1}{n}\sum_{i=1}^n{\log{P(\text{y}=y_i|x_i;\theta)}}
\end{aligned} \\
\text{where }P(\text{y}=\tilde{y}|\text{x}=x_i;\theta)=y_i\odot\hat{y}_i=.1 \\
\text{ and }\tilde{y}=\underset{y\in\mathcal{Y}}{\text{argmax }}P(\text{y}=y|\text{x}=x_i).
\end{gathered}$$

따라서 우리는 크로스 엔트로피를 최소화(minimize)하는 것이 log-likelihood를 최대화(maximize)하는 것을 확인 할 수 있습니다.
