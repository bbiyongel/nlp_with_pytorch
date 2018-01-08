# Supervised NMT

## Cross-entropy vs BLEU

$$
L= -\frac{1}{|Y|}\sum_{y \in Y}{P(y) \log P_\theta(y)}
$$

Cross entropy는 훌륭한 classification을 위해서 이미 훌륭한 loss function이지만 약간의 문제점을 가지고 있습니다. Seq2seq의 훈련 과정에 적용하게 되면, 그 자체의 특성으로 인해서 우리가 평가하는 BLEU와의 괴리가 생기게 됩니다. (자세한 내용은 이전 챕터 내용 참조) 따라서 어찌보면 우리가 원하는 번역 task의 objective와 다름으로 인해서 cross-entropy 자체에 over-fitting되는 효과가 생길 수 있습니다. 일반적으로 BLEU는 implicit evluation과 좋은 상관관계에 있다고 알려져 있지만, cross entropy는 이에 비해 낮은 상관관계를 가지기 때문입니다. 따라서 차라리 BLEU를 objective function으로 사용하게 된다면 더 좋은 결과를 얻을 수 있습니다.

## Minimum Risk Training

위의 아이디어에서 출발한 논문[\[Shen at el.2015\]](https://arxiv.org/pdf/1512.02433.pdf)이 Minimum Risk Training이라는 방법을 제안하였습니다. 이때에는 Policy Gradient를 직접적으로 사용하진 않았지만, 거의 비슷한 방법으로 policy gradient가 정립되기 이전에 제안된 방법이라는 점이 인상적입니다.

$$
\hat{\theta}_{MLE} = argmax_\theta(\mathcal{L}(\theta))
$$

$$
where~\mathcal{L}(\theta)=\sum_{s=1}^S\log{P(y^{(s)}|x^{(s)};\theta)}
$$

기존의 Maximum Likelihood Estimation (MLE)방식의 위와 같은 Loss function을 사용하여 $$ |S| $$개의 입력과 출력에 대해서 loss 값을 구하고, 이를 최대화 하는 $$ \theta $$를 찾는 것이 objective였습니다. 하지만 이 논문에서는 ***Risk***를 아래와 같이 정의하고, 이를 최소화 하는 것을 Minimum Risk Training (MRT)라고 하였습니다.

$$
\mathcal{R}(\theta)=\sum_{s=1}^S E_{y|x^{(s)};\theta}[\triangle(y,y^{(s)})]
$$

$$
=\sum_{s=1}^S \sum_{y \in \mathcal{Y(x^{(s)})}}{P(y|x^{(s)};\theta) \triangle(y, y^{(s)})}
$$

위의 수식에서 $$ \mathcal{Y}(x^{(s)}) $$는 full search space로써, $$ s $$번째 입력 $$ x^{(s)} $$가 주어졌을 때, 가능한 정답의 집합을 의미합니다. 또한 $$ \triangle(y,y^{(s)}) $$는 주어진 입력과 파라미터($$ \theta $$)가 있을 때, sampling한 $$ y $$와 실제 정답 $$ y^{(s)} $$의 차이값을 나타냅니다. 즉, 위 수식에 따르면 ***risk*** $$ \mathcal{R} $$은 주어진 입력과 현재 파라미터 상에서 sampling한 y를 통해 현재 모델(함수)을 approximation하고, 동시에 이를 사용하여 ***risk***의 기대값을 구한다고 볼 수 있습니다.

$$
\hat{\theta}_{MRT}=argmin_\theta(\mathcal{R}(\theta))
$$

이렇게 정의된 ***risk***를 최소화 하도록 하는 것이 objective입니다. 따라서 실제 구현에 있어서는 $$ \triangle(y,y^{(s)}) $$ 사용을 위해서 BLEU 점수에 $$ -1 $$을 곱하여 사용하기도 합니다.

$$
\tilde{\mathcal{R}}(\theta)=\sum_{s=1}^S{E_{y|x^{(s)};\theta,\alpha}[\triangle(y,y^{(s)})]}
$$
$$
=\sum_{s=1}^S \sum_{y \in \mathcal{S}(x^{(s)})}{Q(y|x^{(s)};\theta,\alpha)\triangle(y,y^{(s)})}
$$
$$
where~\mathcal{S}(x^{(s)})~is~a~sampled~subset~of~the~full~search~space~\mathcal{y}(x^{(s)})
$$
$$
and~Q(y|x^{(s)};\theta,\alpha)~is~a~distribution~defined~on~the~subspace~S(x^{(s)}):
$$
$$
Q(y|x^{(s)};\theta,\alpha)=\frac{P(y|x^{(s)};\theta)^\alpha}{\sum_{y' \in S(x^{(s)})}P(y'|x^{(s)};\theta)^\alpha}
$$

![](/assets/rl-minimum-risk-training.png)

## RL in GNMT

## Policy Gradient for NMT

[\[Gu at el.2017\]](https://arxiv.org/pdf/1702.02429.pdf)

https://arxiv.org/pdf/1707.07402.pdf

https://arxiv.org/pdf/1607.07086.pdf

https://arxiv.org/pdf/1602.01783.pdf

https://arxiv.org/pdf/1505.00521.pdf
