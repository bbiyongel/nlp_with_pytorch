# 언어모델의 평가 방법

좋은 언어모델이란 실제 우리가 쓰는 언어에 대해서 최대한 비슷하게 확률 분포를 근사하는 모델(또는 파라미터, $\theta$)이 될 것입니다. 많이 쓰이는 문장 또는 표현일수록 높은 확률을 가져야 하며, 적게 쓰이는 문장 또는 이상한 문장 또는 표현일수록 확률은 낮아야 합니다. 즉, 우리가 (공들여) 준비한 테스트 문장을 잘 예측 해 낼 수록 좋은 언어모델이라고 할 수 있습니다. 문장을 잘 예측 한다는 것은, 다르게 말하면 주어진 테스트 문장이 언어모델에서 높은 확률을 가진다고 할 수 있습니다. 또는 문장의 앞부분이 주어지고, 다음에 나타나는 단어의 확률 분포가 실제 테스트 문장의 다음 단어에 대해 높은 확률을 갖는다면 더 좋은 언어모델이라고 할 수 있을 것 입니다. 이번 섹션은 언어모델의 성능을 평가하는 방법에 대해서 다루도록 하겠습니다.

## Perplexity

Perplexity(퍼플렉시티, PPL) 측정 방법은 정량 평가(explicit evaluation) 방법의 하나입니다. PPL을 이용하여 언어모델 상에서 테스트 문장들의 점수를 구하고, 이를 기반으로 언어모델의 성능을 측정합니다. PPL은 문장의 확률에 길이에 대해서 normalization한 값이라고 볼 수 있습니다.

$$
\begin{aligned}
\text{PPL}(w_1,w_2,\cdots,w_n)=&P(w_1,w_2,\cdots,w_n)^{-\frac{1}{n}} \\
=&\sqrt[n]{\frac{1}{P(w_1,w_2,\cdots,w_n)}}
\end{aligned}
$$

문장이 길어지게 되면 문장의 확률은 굉장히 작아지게 됩니다. 이는 체인룰에 따라서 조건부 확률들의 곱으로 바꿔 표현하여 보면 알 수 있습니다. 따라서 우리는 문장의 길이 n으로 제곱근을 취해 기하평균을 구하고, 문장 길이에 대해서 normalization을 해 주는 것을 볼 수 있습니다. 문장의 확률이 분모에 들어가 있기 때문에, 확률이 높을수록 PPL은 작아지게 됩니다.

따라서 테스트 문장에 대해서 확률을 높게 예측할 수록 좋은 언어모델인 만큼, 해당 테스트 문장에 대한 PPL이 작을 수록 좋은 언어모델이라고 할 수 있습니다. 즉, PPL은 수치가 낮을수록 좋습니다. 또한, n-gram의 n이 클 수록 보통 더 낮은 PPL을 보여주기도 합니다.

위 PPL의 수식은 다시한번 체인룰에 의해서

$$
=\sqrt[n]{\frac{1}{\prod_{i=1}^{n}{P(w_i|w_1,\cdots,w_{i-1})}}}
$$

라고 표현 될 수 있고, 여기에 n-gram이 적용 될 경우,

$$
\approx\sqrt[n]{\frac{1}{\prod_{i=1}^{n}{P(w_i|w_{i-n+1},\cdots,w_{i-1})}}}
$$

로 표현 될 수 있습니다.

## Perplexity의 해석

![주사위 두 개](../assets/lm_rolling_dice.png)

<stop>

Perplexity(PPL)의 개념을 좀 더 짚고 넘어가도록 해 보겠습니다. 예를 들어 우리가 6면 주사위를 던져서 나오는 값을 통해 수열을 만들어낸다고 해 보겠습니다. 따라서 1부터 6까지 숫자의 출현 확률은 모두 같다(uniform distribution)고 가정하겠습니다. 그럼 N번 주사위를 던져 얻어내는 수열에 대한 perplexity는 아래와 같습니다.

$$
\text{PPL}(x)=({\frac{1}{6}}^{N})^{-\frac{1}{N}}=6
$$

매 time-step 가능한 경우의 수인 6이 PPL로 나왔습니다. 즉, PPL은 우리가 뻗어나갈 수 있는 branch(가지)의 숫자를 의미하기도 합니다. 다른 예를 들어 만약 20,000개의 어휘로 이루어진 뉴스 기사에 대해서 PPL을 측정한다고 하였을 때, 단어의 출현 확률이 모두 동일하다면 PPL은 20,000이 될 것입니다. 하지만 3-gram을 사용한 언어모델을 만들어 측정한 PPL이 30이 나왔다면, 우리는 이 언어모델을 통해 해당 신문기사에서 매번 기사의 앞 부분을 통해 다음 단어를 예측 할 때, 평균적으로 30개의 후보 단어 중에서 선택할 수 있다는 얘기가 됩니다. 따라서 우리는 perplexity를 통해서 언어모델의 성능을 단순히 측정할 뿐만 아니라 실제 어느정도인지 가늠 해 볼 수도 있습니다.

## 엔트로피와 Perplexity의 관계

우리는 앞서 책 초반부에서 엔트로피에 대해서 다루었습니다. 엔트로피는 정보량의 평균을 의미하였고, 정보량이 낮으면 확률 분포는 날카로운(sharp) 모양을 하여 확률이 높았으며, 반대로 정보량이 높으면 확률 분포는 납작(flat)해진다고 하였습니다.

크로스 엔트로피의 수식은 아래와 같습니다.

$$
H(P, P_\theta)=-\sum_{x\in\mathcal{X}}{P(x)\log{P_\theta(x)}}
$$

이를 실제 ground-truth 언어모델 $P(\text{x})$로부터, 우리가 가진 확률 분포 $P_\theta(\text{x})$에 대한 크로스 엔트로피 손실 함수로 표현해 보겠습니다.

$$
\mathcal{L}(P, P_\theta)=-\sum_{w\in\mathcal{W}}{P(w)\log{P_\theta(w)}}
$$

크로스 엔트로피 함수를 읽어보면, 모든 가능한 문장의 집합 $\mathcal{W}$로부터 샘플링한 $w$를 통해 적분(합) 하는 것을 볼 수 있습니다. 그럼 길이 n의 문장 한 개에 대한 정보량은 아래와 같이 표현 가능할 것 입니다.

$$
\begin{aligned}
\mathcal{L}&=-P(w_1^n)\log{P_\theta(w_1^n)} \\
&\approx-\frac{1}{N}\sum_{i=1}^{N}{\log{P_\theta(w_i|w_{<i})}} \\
&=\log{\Big(\prod_{i=1}^{N}{P_\theta(w_i|w_{<i})}\Big)^{-\frac{1}{N}}} \\
&=\log{\sqrt[N]{\frac{1}{\prod_{i=1}^{N}{P_\theta(w_i|w_{<i})}}}} \\
\end{aligned}
$$

여기에 PPL 수식을 다시 떠올려 보겠습니다.

$$
\begin{gathered}
\text{PPL}(W)=P(w_1, w_2, \cdots, w_N)^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_1,w_2,\cdots,w_N)}} \\
\text{by chain rule},\\
\text{PPL}(W)=\sqrt[N]{\prod_{i=1}^{N}{\frac{1}{P(w_i|w_1,\cdots,w_{i-1})}}}
\end{gathered}
$$

앞서 정리했던 Cross Entropy와 수식이 비슷한 형태임을 알 수 있습니다. 따라서 PPL과 Cross Entropy의 관계는 아래와 같습니다.

$$
\text{PPL}=\exp(\text{Cross Entropy})
$$

따라서, 우리는 Maximum Likelihood Estimation(MLE)을 통해 parameter($\theta$)를 배울 때, cross entropy를 통해 얻은 ($P_\theta$의 로그 확률 값) loss 값에 $\exp$를 취함으로써, perplexity를 얻어 언어모델의 성능을 나타낼 수 있습니다.
