# 나이브 베이즈를 활용하기

나이브 베이즈(Naive Bayes)는 간단하지만 매우 강력한 방법 입니다. 의외로 기대 이상의 성능을 보여줄 때가 많습니다. 물론 단어를 여전히 discrete한 심볼(symbol)로 다루기 때문에, 여전히 아쉬운 부분이 많습니다. 이번 섹션에서는 나이브 베이즈를 통해서 텍스트를 분류 하는 방법을 살펴 보겠습니다.

## 사후 확률 최대화 (Maximum A Posterior, MAP)

나이브 베이즈를 소개하기에 앞서, 베이즈 정리(Bayes Theorem)을 짚고 넘어가지 않을 수 없습니다. 먼저 우리가 알고자 하는 것은 데이터 $\mathcal{D}$가 주어졌을 때, 각 클래스 c의 확률 입니다.

$$
P(c|\mathcal{D})
$$

이 확률 함수를 베이즈 정리를 활용하여 아래와 같이 바꿀 수 있을 것 입니다. Thomas Bayes(토마스 베이즈)가 정립한 이 정리에 따르면 조건부 확률은 아래와 같이 표현 될 수 있습니다.

$$
\begin{aligned}
\underbrace{P(c|\mathcal{D})}_{posterior}&=\frac{\overbrace{P(\mathcal{D}|c)}^{likelihood}\overbrace{P(c)}^{prior}}{\underbrace{P(\mathcal{D})}_{evidence}} \\
&=\frac{P(\mathcal{D}|c)P(c)}{\sum_{i=1}^{|\mathcal{C}|}{P(\mathcal{D}|c_i)P(c_i)}}
\end{aligned}
$$

그리고 각 부분은 명칭을 갖고 있습니다.

|수식|영어 명칭|한글 명칭|
|-|-|-|
|P(c\|D)|Posterior|사후 확률|
|P(D\|c)|Likelihood|가능도(우도)|
|P(c)|Prior|사전 확률|
|P(D)|Evidence|증거|

우리가 풀고자하는 대부분의 문제들은 $P(\mathcal{D})$는 구하기 힘들기 때문에, 보통은 아래와 같이 접근 하기도 합니다. <comment> 또한 우리가 원하는 $P(c|\mathcal{D})$ 자체는 클래스 c에 관한 함수입니다. </comment>

$$
P(c|\mathcal{D}) \varpropto P(\mathcal{D}|c)P(c)
$$

위의 성질을 이용하여 주어진 데이터 $\mathcal{D}$를 만족하며 확률을 최대로 하는 클래스 $c$를 구할 수 있습니다. 이처럼 posterior 확률을 최대화(maximize)하는 클래스 $c$를 구하는 것을 Maximum A Posterior (MAP)라고 부릅니다. 그 수식은 아래와 같습니다.

$$
\hat{c}_{\text{MAP}}=\underset{c\in\mathcal{C}}{\text{argmax }}P(C=c|\mathcal{D})
$$

다시 한번 수식을 살펴보면, $\mathcal{D}$(데이터)가 주어졌을 때, 가능한 클래스의 집합 $\mathcal{C}$ 중에서 사후 확률을 최대로 하는 클래스 $c$를 선택하는 것 입니다.

이와 마찬가지로 데이터 $\mathcal{D}$가 나타날 라이클리후드(likelihood) 확률을 최대로 하는 클래스 $c$를 선택하는 것을 Maximum Likelihood Estimation (MLE)라고 합니다. <comment> 우리는 MLE에 대해서 앞선 챕터에서 살펴보았습니다. </comment>

$$
\hat{c}_{\text{MLE}}=\underset{c\in\mathcal{C}}{\text{argmax }}P(\mathcal{D}|C=c)
$$

MLE는 주어진 데이터 $\mathcal{D}$와 클래스 레이블(label) $C$가 있을 때, 확률 분포를 근사하기 위한 함수 파라미터 $\theta$를 훈련하는 방법으로 사용 됩니다.

$$
\hat{\theta}=\underset{\theta}{\text{argmax }}P(C|\mathcal{D},\theta)
$$

### MLE vs MAP

경우에 따라 MAP는 MLE에 비해서 좀 더 정확할 수 있습니다. 사전(prior)확률이 반영되어 있기 때문 입니다. 예를 들어, 만약 범죄현장에서 발자국을 발견하고 사이즈를 측정했더니 범인은 신발사이즈(데이터, $\text{x}$) 155를 신는 사람인 것으로 의심 됩니다. 이때, 범인의 성별(클래스, $\text{y}$)을 예측 해 보도록 하죠.

이때, 성별 클래스의 집합은 $Y=\{\text{male}, \text{female}\}$ 입니다. 신발사이즈 $X$는 5단위의 정수로 이루어져 있습니다. $X=\{\cdots,145,150,155,160,\cdots\}$. 신발사이즈 155는 남자 신발사이즈 치곤 매우 작은 편 입니다. 따라서 우리는 보통 범인을 여자라고 특정할 것 같습니다. 다시 말하면, 남자일 때 신발사이즈 155일 확률 $P(\text{x}=155|\text{y}=\text{male})$은 여자일 때 신발사이즈 155일 확률 $P(\text{x}=155|\text{y}=\text{female})$일 확률 보다 낮습니다.

보통의 경우 남자와 여자의 비율은 $0.5$로 같기 때문에, 이는 큰 상관이 없는 예측 입니다. 하지만 범죄 현장이 만약 남자들이 대부분인 군부대였다면 어떻게 될까요? 남녀 성비는 $P(\text{y}=\text{male}) > P(\text{y}=\text{female})$로 매우 불균형 할 것입니다. 이때, 이미 갖고 있는 라이클리후드에 사전 확률(prior)을 곱해주면 사후확률(posterior)을 최대화 하는 클래스를 더 정확하게 예측 할 수 있습니다.

$$
\begin{gathered}
P(\text{y}=\text{male}|\text{x}=155) > P(\text{y}=\text{female}|\text{x}=155), \\
\text{if }P(\text{x}=155|\text{y}=\text{male})P(\text{y}=\text{male}) > P(\text{x}=155|\text{y}=\text{female})P(\text{y}=\text{female}).
\end{gathered}
$$

## 나이브 베이즈

나이브 베이즈(Naive Bayes)는 MAP를 기반으로 동작합니다. 대부분의 경우 사후확률을 바로 구하기 어렵기 때문에, 라이클리후드와 사전확률의 곱을 통해 클래스 $\text{y}$를 예측 합니다. 먼저 우리가 알고자 하는 값인 사후확률은 아래와 같을 것 입니다.

$$
P(\text{y}=c|\text{x}=w_1,w_2,\cdots,w_n)
$$

이때, $\text{x}$가 다양한 특징(feature)들로 이루어진 데이터라면, 훈련 데이터에서 매우 희소(rare)할 것이므로 사후확률 뿐만 아니라, 라이클리후드 $P(\text{x}=w_1,w_2,\cdots,w_n|\text{y}=c)$를 구하기 어려울 것 입니다. 왜냐면 보통 확률을 코퍼스의 출현빈도를 통해 추정할 수 있는데, 피쳐가 복잡할 수록 라이클리후드 또는 사후확률을 만족하는 경우는 코퍼스에 매우 드물 것이기 때문입니다. 그렇다고 코퍼스에 없는 피쳐의 조합이라고 해서 확률 값을 0으로 추정하는 것은 너무나 강력한 가정이 될 것 입니다.

이때 나이브 베이즈가 강력한 힘을 발휘 합니다. 각 피쳐들이 독립 이라고 가정하는 것 입니다. 그럼 각 피쳐들의 결합 확률(joint probability)을 각 독립된 확률의 곱으로 근사(approximate)할 수 있습니다. 이 과정을 수식으로 표현하면 아래와 같습니다.

$$
\begin{aligned}
P(\text{y}=c|\text{x}=w_1,w_2,\cdots,w_n) &\varpropto P(\text{x}=w_1,w_2,\cdots,w_n|\text{y}=c)P(\text{y}=c) \\
&\approx P(w_1|c)P(w_2|c)\cdots P(w_n|c)P(c) \\
&=\prod_{i=1}^{n}{P(w_i|c)}P(c)
\end{aligned}
$$

따라서, 우리가 구하고자 하는 MAP를 활용한 클래스는 아래와 같이 사후 확률을 최대화하는 클래스가 되고, 이는 나이브 베이즈의 가정에 따라 각 피쳐들의 확률의 곱에 사전 확률을 곱한 값을 최대화 하는 클래스와 같을 것 입니다.

$$
\begin{aligned}
\hat{c} &= \underset{c \in \mathcal{C}}{\text{argmax }}{P(\text{y}=c|\text{x}=w_1,w_2,\cdots,w_n)} \\
&\approx\underset{c \in \mathcal{C}}{\text{argmax }}{\prod_{i=1}^{n}{P(w_i|c)}P(c)}
\end{aligned}
$$

이때 사용되는 사전확률은 아래와 같이 실제 데이터(코퍼스)에서 출현한 빈도를 통해 추정할 수 있습니다.

$$
P(\text{y}=c)\approx\frac{\text{Count}(c)}{\sum_{i=1}^{|\mathcal{C}|}{\text{Count}(c_i)}}
$$

또한, 각 특징 별 라이클리후드 확률도 데이터에서 바로 구할 수 있습니다. 만약 모든 피쳐들의 조합이 데이터에서 나타난 횟수를 통해 확률을 구하려 하였다면 희소성(sparseness) 문제 때문에 구할 수 없었을 것 입니다. 하지만 나이브 베이즈의 가정(각 피쳐들은 독립)을 통해서 쉽게 데이터 또는 코퍼스에서 출현 빈도를 활용할 수 있게 되었습니다.

$$
P(w|c)\approx\frac{\text{Count}(w,c)}{\sum_{j=1}^{|V|}{\text{Count}(w_j,c)}}
$$

이처럼 간단한 가정을 통하여 데이터의 희소성(sparsity) 문제를 해소하여, 간단하지만 강력한 방법으로 우리는 사후 확률를 최대화하는 클래스를 예측 할 수 있게 되었습니다.

## 예제: 감성분석 (Sentiment Analysis)

그럼 실제 예제로 접근해 보죠. 감성분석은 가장 많이 활용되는 텍스트 분류 기법 입니다. 사용자의 댓글이나 리뷰 등을 긍정 또는 부정으로 분류하여 마케팅이나 서비스 향상에 활용하고자 하는 방법 입니다.

$$
\begin{gathered}
\mathcal{C}=\{\color{cyan}\text{pos}\color{default},\color{red}\text{neg}\color{default}\} \\
\mathcal{D}=\{d_1,d_2,\cdots\}
\end{gathered}
$$

위와 같이 긍정($\color{cyan}\text{pos}$)과 부정($\color{red}\text{neg}$)으로 클래스의 집합 $\mathcal{C}$가 구성되어 있고, 문서 $d$로 구성된 데이터 $\mathcal{D}$가 있습니다. 이때, 우리에게 "I am happy to see this movie!"라는 문장이 주어졌을 때, 이 문장이 긍정인지 부정인지 판단해 보겠습니다.

$$
\begin{aligned}
P(\color{cyan}\text{pos}\color{default}|\text{I, am, happy, to, see, this, movie, !})&= \frac{P(\text{I, am, happy, to, see, this, movie, !}|\color{cyan}\text{pos}\color{default})P(\color{cyan}\text{pos}\color{default})}{P(\text{I, am, happy, to, see, this, movie, !})}\\
&\approx \frac{P(I|\color{cyan}\text{pos}\color{default})P(am|\color{cyan}\text{pos}\color{default})P(happy|\color{cyan}\text{pos}\color{default})\cdots P(!|\color{cyan}\text{pos}\color{default})P(\color{cyan}\text{pos}\color{default})}{P(\text{I, am, happy, to, see, this, movie, !})}
\end{aligned}
$$

나이브 베이즈를 활용하여 단어의 조합에 대한 확률을 각각 분해할 수 있습니다. 즉, 우리는 각 단어의 출현 확률을 독립이라고 가정 한 이후에, 결합 라이클리후드 확률을 모두 각각의 라이클리후드 확률로 분해합니다. 그리고 그 확률들은 아래와 같이 데이터 $\mathcal{D}$에서의 출현 빈도를 통해 구할 수 있습니다.

$$
\begin{aligned}
P(\text{happy}|\color{cyan}\text{pos}\color{default})&\approx\frac{\text{Count}(\text{happy},\color{cyan}\text{pos}\color{default})}{\sum_{j=1}^{|V|}{\text{Count}(w_j,\color{cyan}\text{pos}\color{default})}} \\
P(\color{cyan}\text{pos}\color{default})&\approx\frac{\text{Count}(\color{cyan}\text{pos}\color{default})}{|\mathcal{D}|}
\end{aligned}
$$

마찬가지로 부정 감성에 대해 같은 작업을 반복 할 수 있습니다.

$$
\begin{aligned}
P(\color{red}\text{neg}\color{default}|\text{I, am, happy, to, see, this, movie, !})&= \frac{P(\text{I, am, happy, to, see, this, movie, !}|\color{red}\text{neg}\color{default})P(\color{red}\text{neg}\color{default})}{P(\text{I, am, happy, to, see, this, movie, !})}\\
&\approx \frac{P(I|\color{red}\text{neg}\color{default})P(am|\color{red}\text{neg}\color{default})P(happy|\color{red}\text{neg}\color{default})\cdots P(!|\color{red}\text{neg}\color{default})P(\color{red}\text{neg}\color{default})}{P(\text{I, am, happy, to, see, this, movie, !})} \\
\\
P(\text{happy}|\color{red}\text{neg}\color{default})&\approx\frac{\text{Count}(\text{happy},\color{red}\text{neg}\color{default})}{\sum_{j=1}^{|V|}{\text{Count}(w_j,\color{red}\text{neg}\color{default})}} \\
P(\color{red}\text{neg}\color{default})&\approx\frac{\text{Count}(\color{red}\text{neg}\color{default})}{|\mathcal{D}|}
\end{aligned}
$$

이처럼 우리는 단순히 코퍼스에서 각 단어의 클래스 당 출현 빈도를 계산하는 것만으로도 간단하게 감성분석을 수행 할 수 있습니다.

## Add-one 스무딩(Smoothing)

여기에 문제가 하나 있습니다. 만약 훈련 데이터에서 $\text{Count}(\text{happy},\color{red}\text{neg}\color{default})$가 0이었다면 $P(\text{happy}|\color{red}\text{neg}\color{default})=0$이 되겠지만, 그저 훈련 데이터(코퍼스)에 존재하지 않는 경우라고 해서, 해당 샘플의 실제 출현 확률을 0으로 추정하는 것은 매우 위험한 일 입니다.

$$
\begin{gathered}
P(\text{happy}|\color{red}\text{neg}\color{default})\approx\frac{\text{Count}(\text{happy},\color{red}\text{neg}\color{default})}{\sum_{j=1}^{|V|}{\text{Count}(w_j,\color{red}\text{neg}\color{default})}}=0, \\
\\
\text{where }\text{Count}(\text{happy},\color{red}\text{neg}\color{default})=0.
\end{gathered}
$$

따라서 우리는 이런 경우를 위하여 각 출현횟수에 1을 더해주어 간단하게 문제를 완화 할 수 있습니다. 물론 완벽한 해결법은 아니지만 나이브 베이즈의 기존 가정과 마찬가지로 매우 간단하고 강력합니다.

$$
\tilde{P}(w|c)=\frac{\text{Count}(w,c)+1}{\big(\sum_{j=1}^{|V|}{\text{Count}(w_j,c)}\big)+|V|}
$$

## 장점과 한계

위와 같이 나이브 베이즈를 통해서 단순히 출현 빈도를 세는 것처럼 쉽고 간단하지만 강력하게 감성분석을 구현 할 수 있습니다. 하지만 문장 "I am not happy to see this movie!"라는 문장이 주어지면 어떻게 될까요? "not"이 추가 되었을 뿐이지만 문장의 뜻은 반대가 되었습니다.

$$
\begin{gathered}
P(\color{cyan}\text{pos}\color{default}|\text{I, am, not, happy, to, see, this, movie, !}) \\
P(\color{red}\text{neg}\color{default}|\text{I, am, not, happy, to, see, this, movie, !})
\end{gathered}
$$

"not"은 "happy"를 수식하기 때문에 두 단어를 독립적으로 보는 것은 옳지 않을 것 입니다.

$$
P(\text{not, happy}) \neq P(\text{not})P(\text{happy})
$$

사실 문장은 단어들이 순서대로 나타나서 의미를 이루기 때문에, 각 단어의 출현 여부도 중요하지만, 각 단어 사이의 순서로 인해 생기는 관계와 정보도 무시할 수 없습니다. 하지만 나이브 베이즈의 가정<comment> 각 피쳐는 서로 독립이다. </comment>은 언어의 이런 특징을 단순화하여 접근하기 때문에 한계가 있습니다. 하지만, 레이블(labeled) 데이터가 매우 적은 경우에는 오히려 복잡한 딥러닝보다 이런 간단한 방법을 사용하는 것이 훨씬 더 나은 대안이 될 수도 있습니다.