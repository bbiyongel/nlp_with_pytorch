# 듀얼리티를 활용한 비지도학습

## 듀얼러닝 기계번역 (Dual Learning for Machine Translation)
  
공교롭게도 CycleGAN과 비슷한 시기에 나온 논문[[Xia at el.2016]](https://arxiv.org/pdf/1611.00179.pdf)이 있습니다. 자연어처리의 특성상 CycleGAN처럼 직접적으로 그래디언트를 전달해 줄 수는 없었지만 기본적으로는 아주 비슷한 개념입니다. 병렬 코퍼스를 활용하여 훈련된 기본 성능의 기게 번역 모델을 단방향 코퍼스를 이용하여 성능을 극대화 하고자 하였습니다. 그래디언트를 GAN과 같이 전달 해 줄 수 없었기 때문에, 강화학습을 활용하여 디스크리미네이터(discriminator)의 값을 전달 해줘야 합니다.

즉, 단방향 코퍼스로부터 받은 문장 $s$ 에 대해서 번역을 하고, 번역 된 문장 $s_{mid}$ 을 사용하여 반대방향 번역을 통해 복원을 하였을 때, 복원 된 문장 $\hat{s}$ 이 원래의 처음 문장과의 차이 $\triangle(\hat{s}, s)$ 가 최소화 되도록 훈련하는 것입니다. 이때, 번역된 문장 $s_{mid}$ 는 자연스러운 해당 언어의 문장이 되었는가도 중요한 지표가 됩니다. CycleGAN과 너무나도 닮아있는 것을 볼 수 있습니다.

![기계번역 듀얼러닝의 알고리즘](../assets/rl-dual-learning-1.png)

위의 알고리즘을 따라가 보겠습니다. $\text{Language }A,\text{ Language }B$ , 두 개의 도메인의 문장들이 주어질 겁니다. 제너레이터 $G_{A \rightarrow B}$ 의 파라미터 $\theta_{AB}$ 와 반대방향 제너레이터 $F_{B \rightarrow A}$ 의 파라미터 $\theta_{BA}$ 가 등장합니다. 이 $G_{A \rightarrow B}, F_{B \rightarrow A}$ 는 모두 병렬 코퍼스를 활용하여 pre-training이 되어 있는 상태 입니다. 우리는 앞서 배운 폴리시그래디언트를 활용하여 파라미터 업데이트를 수행할 수 있습니다.

$$\begin{aligned}
\theta_{AB}\leftarrow\theta_{AB}+\gamma\triangledown_{\theta_{AB}}\hat{\mathbb{E}}[r] \\
\theta_{BA}\leftarrow\theta_{BA}+\gamma\triangledown_{\theta_{BA}}\hat{\mathbb{E}}[r]
\end{aligned}$$

 $\hat{\mathbb{E}}[r]$ 을 각각의 파라미터에 대해서 미분 해 준 값을 더해주는 것을 볼 수 있습니다. 이 보상(reward)의 기대값은 아래와 같이 구할 수 있습니다.

$$\begin{gathered}
r=\lambda\cdot{r_{AB}}+(1-\lambda)\cdot{r_{BA}}, \\
\\
\text{where }r_{AB}=\text{LM}_{B}(s_{mid}) \\
\text{and }r_{BA}=\log{P(s|s_{mid};\theta_{BA})}. \\
\end{gathered}$$

$k$ 개의 샘플링한 문장에 대해서 각기 방향에 대한 보상을 각각 구한 후, 이를 선형 결합(linear combination) 취해줍니다. 이때 $s_{mid}$ 는 샘플링한 문장을 의미하고, $\text{LM}_B$ 를 사용하여 해당 문장이 $\text{Language }B$ 의 집합에 잘 어울리는지 보상으로 리턴합니다. 언어모델 $\text{LM}_B$ 는 기존의 $\text{Language }B$ 의 단방향 코퍼스를 활용하여 pre-training 되어 있기 때문에, 자연스러운 문장이 생성되었을수록 해당 언어모델에서 높은 확률을 가질 수 있을 겁니다. 우리는 다수의 단방향 코퍼스를 갖고 있기 때문에 언어모델을 만들어내는 것은 어려운 일이 아닙니다.

$$\begin{aligned}
\triangledown_{\theta_{AB}}\hat{\mathbb{E}}[r]&=\frac{1}{K}\sum_{k=1}^K{[r_k\triangledown_{\theta_{AB}}\log{P(s_{mid,k}|s;\theta_{AB})}]} \\
\triangledown_{\theta_{BA}}\hat{\mathbb{E}}[r]&=\frac{1}{K}\sum_{k=1}^K[(1-\lambda)\triangledown_{\theta_{BA}}\log{P(s|s_{mid,k};\theta_{BA})}]
\end{aligned}$$

이렇게 얻어진 $\mathbb{E}[r]$ 를 각 파라미터에 대해서 미분하게 되면 위와 같은 수식을 얻을 수 있고, 상기 서술한 파라미터 업데이트 수식에 대입하면 됩니다. 비슷한 방식으로 $B \rightarrow A$ 를 구할 수 있습니다.

|모델|En $\rightarrow$ Fr|Fr $\rightarrow$ En|En $\rightarrow$ Fr (병렬코퍼스 최소화)|Fr $\rightarrow$ En (병렬코퍼스 최소화)|
|-|-|-|-|-|
|NMT|29.92|27.49|25.32|22.27|
|NMT + Back Translation|30.40|27.66|25.63|23.24|
|듀얼러닝|32.06|29.78|28.73|27.50|

<!--
![](../assets/rl-dual-learning-2.png)
-->

![듀얼러닝의 적용 결과](../assets/rl-dual-learning-3.png)

위 그래프에서 문장의 길이와 상관 없이 모든 구간에서 베이스라인 NMT를 성능으로 압도하고 있는 것을 알 수 있습니다. 다만, 병렬 코퍼스의 양이 커질수록 단방향코퍼스를 활용한 듀얼러닝에 의한 성능 향상의 폭이 줄어드는 것을 확인 할 수 있습니다. 이 방법은 강화학습과 듀얼리티를 접목하여 적은 양의 병렬 코퍼스와 다수의 단방향 코퍼스를 활용하여 번역기의 성능을 효과적으로 끌어올리는 방법을 제시하였다는 점에서 주목할 만 합니다.

## Dual Unsupervised Learning with Marginal Distribution Regularization

앞서 설명한 Dual Supervised Learning (DSL)은 베이즈 정리에 따른 수식을 제약조건으로 사용하였다면, 이 방법[[Wang et al.2017]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/17041-72820-1-SM.pdf)은 Marginal 분포(distribution)의 성질을 이용하여 제약조건을 만듭니다. <comment> Marginal 분포에 대한 설명은 앞서 기초수학 챕터에서 간단하게 이야기한 바 있습니다. </comment>

$$P(y)=\sum_{x \in \mathcal{X}}{P(x,y)}=\sum_{x \in \mathcal{X}}{P(y|x)P(x)}$$

우리는 marginal 분포를 통해 위의 수식이 항상 참(true)임을 알고 있습니다. 이것을 조건부확률로 나타낼 수 있고, 여기서 한발 더 나아가 기대값 표현으로 바꿀 수 있습니다. 그리고 이를 K번 샘플링 하도록 하여 몬테카를로 샘플링으로 근사하여 표현 할 수 있습니다.

$$\begin{aligned}
P(y)=\sum_{x \in \mathcal{X}}{P(y|x;\theta)P(x)}&=\mathbb{E}_{\text{x}\sim{P(\text{x})}}\big[P(y|\text{x};\theta)\big] \\
&\approx\frac{1}{K}\sum^K_{i=1}{P(y|x^i;\theta)},\text{ where }x^i\sim{P(\text{x})}
\end{aligned}$$

이제 위의 수식을 기계번역에 적용해 보도록 하겠습니다. 우리에게 아래와 같이 N 개의 소스(source) 문장 $x$ , 타겟(target) 문장 $y$ 으로 이루어진 양방향 병렬 코퍼스 $\mathcal{B}$ , S 개의 타겟 문장 $y$ 로만 이루어진 단방향 코퍼스 $\mathcal{M}$ 이 있다고 가정 해 보겠습니다.

$$\begin{aligned}
\mathcal{B}&=\{(x^n, y^n)\}^N_{n=1} \\
\mathcal{M}&=\{y^s\}^S_{s=1}
\end{aligned}$$

그럼 우리는 아래의 목적함수(objective function)을 최대화(maximize)하는 동시에 marginal 분포에 따른 제약조건 또한 만족시켜야 합니다.

$$\begin{aligned}
&Objective: \sum^N_{n=1}{\log{P(y^n|x^n;\theta)}}, \\
&\text{s.t. }P(y)=\mathbb{E}_{\text{x}\sim{P(\text{x})}}\big[P(y|\text{x};\theta)\big]\text{, }\forall{y}\in\mathcal{M}.
\end{aligned}$$

위의 수식을 DSL과 마찬가지로 $\lambda$ 와 함께 $S(\theta)$ 와 같이 표현하여 기존의 손실함수(loss function)에 추가하여 줍니다.

$$\begin{gathered}
\mathcal{S}(\theta)=\Big[\log\hat{P}(y)-\log{\mathbb{E}_{\text{x}\sim\hat{P}(\text{x})}\big[P(y|\text{x};\theta)}\big]\Big]^2 \\
\\
\begin{aligned}
\mathcal{L}(\theta)&=-\sum^N_{n=1}{\log{P(y^n|x^n;\theta)}}+\lambda\mathcal{S}(\theta) \\
&=-\sum^N_{n=1}{\log{P(y^n|x^n;\theta)}}+\lambda\sum^S_{s=1}{\Big[\log\hat{P}(y^s)-\log{\mathbb{E}_{\text{x}\sim\hat{P}(\text{x})}\big[P(y^s|\text{x};\theta)}\big]\Big]^2}
\end{aligned}
\end{gathered}$$

이때, DSL과 유사하게 $\hat{P}(x)$ 와 $\hat{P}(y)$ 가 등장합니다. $\hat{P}(y)$ 는 단방향 코퍼스로 만든 언어모델을 통해 계산한 각 문장들의 확률값을 의미합니다. 위의 수식에 따르면 $\hat{P}(x)$ 를 통해 소스(source) 문장 $x$ 를 샘플링하여 네트워크 $\theta$ 를 통과시켜 $P(y|x;\theta)$ 를 구해야겠지만, 아래와 같이 좀 더 다른 방법으로 접근합니다.

$$\begin{aligned}
P(y)=\mathbb{E}_{\text{x}\sim\hat{P}(\text{x})}\big[P(y|x;\theta)\big]&=\sum_{x\in\mathcal{X}}{P(y|x;\theta)\hat{P}(x)} \\
&=\sum_{x\in\mathcal{X}}\frac{P(y|x;\theta)\hat{P}(x)}{P(x|y)}P(x|y) \\
&=\mathbb{E}_{\text{x}\sim\text{P(x|y)}}\Bigg[\frac{P(y|\text{x};\theta)\hat{P}(\text{x})}{P(\text{x}|y)}\Bigg] \\
&\approx\frac{1}{K}\sum^K_{i=1}{\frac{P(y|x_i;\theta)\hat{P}(x_i)}{P(x_i|y)}}\text{, }x_i\sim{P(\text{x}|y)}
\end{aligned}$$

위와 같이 importance sampling을 통해, 타겟 언어의 문장 $y$ 를 반대 방향 번역기( $y\rightarrow{x}$ )에 넣어 K 개의 소스 언어의 문장 $x$ 를 샘플링하여 $P(y)$ 를 구합니다. 이 과정을 다시 하나의 손실함수로 표현하면 아래와 같습니다. 우변의 첫 번째 텀 $\sum^N_{n=1}{\log{P(y^n|x^n;\theta)}}$ 은 문장 $x^n$ 이 주어졌을 때, $y^n$ 의 확률을 최대로 하는 $\theta$ 를 찾도록 합니다. 두 번째 텀은 단방향 코퍼스에서 주어진 문장 $y^s$ 의 언어모델에서의 확률 값 $\log{\hat{P}(y^s)}$ 과의 차이를 줄여야 합니다. 그 값은 반대방향( $\text{y}\rightarrow\text{x}$ ) 번역기를 통해 K번 샘플링한 문장 $x_i$ 의 언어모델 확률값 $\hat{P}(x_i)$ 과 $x_i$ 가 주어졌을 때, $y^s$ 의 확률값을 곱하고, 문장 $y^s$ 가 주어졌을 때 샘플링한 문장 $x_i$ 의 확률값으로 나누어준 값이 됩니다.

$$\mathcal{L}(\theta)\approx-\sum^N_{n=1}{\log{P(y^n|x^n;\theta)}}+\lambda\sum^S_{s=1}{\Bigg[\log{\hat{P}(y^s)}-\log{\frac{1}{K}\sum^K_{i=1}\frac{\hat{P}(x_i)P(y^s|x_i\theta)}{P(x_i|y^s)}}\Bigg]^2}$$

해당 논문에 따르면 Dual Unsupervised Learning을 적용하여 다른 기존의 단방향 코퍼스를 활용하는 알고리즘들과 비교한 결과는 아래의 테이블과 같습니다.

|모델|En $\rightarrow$ Fr| $\triangle$ |DE $\rightarrow$ En| $\triangle$ |
|-|-|-|-|-|
|기본 NMT|29.92||30.99||
|언어모델 앙상블 [Gulcehre et al.2015]|30.03|+0.11|31.08|+0.09|
|Back Translation [Sennrich, Haddow and Birch 2016]|30.40|+0.48|31.76|+0.77|
|듀얼러닝 기계번역 [He et al.2016a]|32.06|+2.14|32.05|+1.06|
|DUL|32.85|+2.93|32.35|+1.36|

<!--
![](../assets/duality-dul-eval.png)
-->

위의 테이블과 같이, 이 방법은 앞 챕터에서 소개한 기존의 단방향 corpus[[Gulcehre et al.2015]](https://arxiv.org/abs/1503.03535)[[Sennrich et al.2016]](https://arxiv.org/abs/1511.06709)를 활용한 방식들과 비교하여 훨씬 더 나은 성능의 개선을 보여주었으며, 바로 앞서 소개한 [Dual Learning[He et al.2016a]](https://arxiv.org/pdf/1611.00179.pdf)보다도 더 나은 성능을 보여줍니다. 마찬가지로, 불안정하고 비효율적인 강화학습을 사용하지 않고도 더 나은 성능을 보여준 것은 주목할 만한 성과라고 할 수 있습니다.

<!--
### 쉬어가기: 임포턴스 샘플링 (Importance Sampling)

Importance 샘플링은 기존의 샘플링하던 분포가 아닌 다른 분포에서 샘플링을 하는 것을 이릅니다. 따라서 윗변과 아랫변에 샘플링하고자 하는 분포 q를 곱해주게 됩니다.

$$\begin{gathered}
\begin{aligned}
\mathbb{E}_{\text{x}\sim{p(\text{x})}}\big[f(\text{x})\big]&=\int_{x}{f(x)p(x)}{dx} \\
&=\int_{x}{\Big( f(x)\frac{p(x)}{q(x)}\Big)\cdot{q(x)}}{dx} \\
&=\mathbb{E}_{\text{x}\sim{q(\text{x})}}\Big[f(\text{x})\frac{p(\text{x})}{q(\text{x})}\Big],
\end{aligned} \\
\forall{q}\text{ (pdf) s.t. }q(x)=0\implies{p(x)=0} \\
\\
w(x)=\frac{p(x)}{q(x)} \\
\begin{aligned}
\mathbb{E}_{\text{x}\sim{q(\text{x})}}\Big[f(\text{x})\frac{p(\text{x})}{q(\text{x})}\Big]&\approx\frac{1}{k}\sum_{i=1}^{k}{f(x_i)\frac{p(x_i)}{q(x_i)}} \\
&=\frac{1}{k}\sum_{i=1}^{k}{f(x_i)w(x_i)} \\
\end{aligned} \\
\text{where }x_i\sim{q(\text{x})}.
\end{gathered}$$
-->
