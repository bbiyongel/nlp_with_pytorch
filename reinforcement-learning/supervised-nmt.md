# Supervised NMT

## Cross-entropy vs BLEU

$$
L= -\frac{1}{|Y|}\sum_{y \in Y}{P(y) \log P_\theta(y)}
$$

Cross entropy는 훌륭한 classification을 위해서 이미 훌륭한 loss function이지만 약간의 문제점을 가지고 있습니다. Seq2seq의 훈련 과정에 적용하게 되면, 그 자체의 특성으로 인해서 우리가 평가하는 BLEU와의 괴리가 생기게 됩니다. (자세한 내용은 이전 챕터 내용 참조) 따라서 어찌보면 우리가 원하는 번역 task의 objective와 다름으로 인해서 cross-entropy 자체에 over-fitting되는 효과가 생길 수 있습니다. 일반적으로 BLEU는 implicit evluation과 좋은 상관관계에 있다고 알려져 있지만, cross entropy는 이에 비해 낮은 상관관계를 가지기 때문입니다. 따라서 차라리 BLEU를 objective function으로 사용하게 된다면 더 좋은 결과를 얻을 수 있습니다. 마찬가지로 다른 NLP task의 문제에 대해서도 비슷한 접근을 생각 할 수 있습니다.

## Minimum Risk Training

위의 아이디어에서 출발한 논문[\[Shen at el.2015\]](https://arxiv.org/pdf/1512.02433.pdf)이 Minimum Risk Training이라는 방법을 제안하였습니다. 이때에는 Policy Gradient를 직접적으로 사용하진 않았지만, 거의 비슷한 수식이 유도 되었다는 점이 매우 인상적입니다.

$$
\begin{aligned}
\hat{\theta}_{MLE} &= argmax_\theta(\mathcal{L}(\theta)) \\
where~\mathcal{L}(\theta)&=\sum_{s=1}^S\log{P(y^{(s)}|x^{(s)};\theta)}
\end{aligned}
$$

기존의 Maximum Likelihood Estimation (MLE)방식의 위와 같은 Loss function을 사용하여 $$ |S| $$개의 입력과 출력에 대해서 loss 값을 구하고, 이를 최대화 하는 $$ \theta $$를 찾는 것이 objective였습니다. 하지만 이 논문에서는 ***Risk***를 아래와 같이 정의하고, 이를 최소화 하는 것을 Minimum Risk Training (MRT)라고 하였습니다.

$$
\begin{aligned}
\mathcal{R}(\theta)&=\sum_{s=1}^S E_{y|x^{(s)};\theta}[\triangle(y,y^{(s)})] \\
&=\sum_{s=1}^S \sum_{y \in \mathcal{Y(x^{(s)})}}{P(y|x^{(s)};\theta) \triangle(y, y^{(s)})}
\end{aligned}
$$

위의 수식에서 $$ \mathcal{Y}(x^{(s)}) $$는 full search space로써, $$ s $$번째 입력 $$ x^{(s)} $$가 주어졌을 때, 가능한 정답의 집합을 의미합니다. 또한 $$ \triangle(y,y^{(s)}) $$는 주어진 입력과 파라미터($$ \theta $$)가 있을 때, sampling한 $$ y $$와 실제 정답 $$ y^{(s)} $$의 차이값을 나타냅니다. 즉, 위 수식에 따르면 ***risk*** $$ \mathcal{R} $$은 주어진 입력과 현재 파라미터 상에서 얻은 y를 통해 현재 모델(함수)을 구하고, 동시에 이를 사용하여 ***risk***의 기대값을 구한다고 볼 수 있습니다.

$$
\hat{\theta}_{MRT}=argmin_\theta(\mathcal{R}(\theta))
$$

이렇게 정의된 ***risk***를 최소화 하도록 하는 것이 objective입니다. 따라서 실제 구현에 있어서는 $$ \triangle(y,y^{(s)}) $$ 사용을 위해서 BLEU 점수에 $$ -1 $$을 곱하여 사용하기도 합니다.

$$
\begin{aligned}
\tilde{\mathcal{R}}(\theta)&=\sum_{s=1}^S{E_{y|x^{(s)};\theta,\alpha}[\triangle(y,y^{(s)})]} \\
&=\sum_{s=1}^S \sum_{y \in \mathcal{S}(x^{(s)})}{Q(y|x^{(s)};\theta,\alpha)\triangle(y,y^{(s)})}
\end{aligned}
$$
$$
\begin{aligned}
where~\mathcal{S}(x^{(s)})~is~a~sampled~subset~of~the~full~search~space~\mathcal{y}(x^{(s)}) \\
and~Q(y|x^{(s)};\theta,\alpha)~is~a~distribution~defined~on~the~subspace~S(x^{(s)}):
\end{aligned}
$$

$$
Q(y|x^{(s)};\theta,\alpha)=\frac{P(y|x^{(s)};\theta)^\alpha}{\sum_{y' \in S(x^{(s)})}P(y'|x^{(s)};\theta)^\alpha}
$$

하지만 주어진 입력에 대한 가능한 정답에 대한 전체 space를 search할 수는 없기 때문에, Monte Carlo를 사용하여 sampling하는 것을 택합니다. 그리고 위의 수식에서 $$ \theta $$에 대해서 미분을 수행합니다.

아래는 위와 같이 훈련한 MRT에 대한 성능을 실험한 결과 입니다. 기존의 MLE 방식에 비해서 BLEU가 상승한 것을 확인할 수 있습니다.

![](/assets/rl-minimum-risk-training.png)

아래에서 설명할 policy gradient 방식과의 **차이점은 sampling을 통해 기대값을 approximation 할 때에 $$ Q $$라는 값을 sampling한 값의 확률들에 대해서 normalize한 형태로 만들어 주었다**는 것 입니다.

## Policy Gradient for GNMT

Google은 GNMT 논문[\[Wo at el.2016\]](https://arxiv.org/pdf/1609.08144.pdf)에서 policy gradient를 사용하여 training criteria를 개량하였습니다.

기존 MLE 방식의 Objective를 아래와 같이 구성합니다. $$ Y^{*(i)} $$은 optimal 정답 데이터를 의미합니다.

$$
\mathcal{O}_{ML}(\theta)=\sum_{i=1}^N\log P_\theta(Y^{*(i)}|X^{(i)})
$$

여기에 추가로 RL방식의 Objective를 추가하였는데 이 방식이 policy gradient 방식과 같습니다.

$$
\mathcal{O}_{RL}(\theta)=\sum_{i=1}^N \sum_{Y \in \mathcal{Y}} P_\theta(Y|X^{(i)})r(Y, Y^{*(i)})
$$

위의 수식도 ***Minimum Risk Training (MRT)*** 방식과 비슷합니다. $$ r(Y, Y^{*(i)}) $$ 또한 정답과 sampling 데이터 사이의 유사도(점수)를 의미합니다. 가장 큰 차이점은 기존에는 risk로 취급하여 minimize하는 방향으로 훈련하였지만, 이번에는 **reward로 취급하여 maximize하는 방향으로 훈련하게 된다는 것 입니다.**

이렇게 새롭게 추가된 objective를 아래와 같이 기존의 MLE방식의 objective와 linear combination을 취하여 최종적인 objective function이 완성됩니다.

$$
\mathcal{O}_{Mixed}(\theta)=\alpha*\mathcal{O}_{ML}(\theta)+\mathcal{O}_{RL}(\theta)
$$

이때에 $$ \alpha $$값은 주로 0.017로 셋팅하였습니다. 위와 같은 방법의 성능을 실험한 결과는 다음과 같습니다.

![](/assets/nmt-gnmt-5.png)

$$ En \rightarrow De $$의 경우에는 성능이 약간 하락함을 보였습니다. 하지만 이는 decoder의 length penalty, coverage penalty와 결합되었기 때문이고, 이 panalty들이 없을 때에는 훨씬 큰 성능 향상이 있었다고 합니다.