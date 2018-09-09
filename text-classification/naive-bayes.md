# Naive Bayes

## Example: Sentiment Analysis

그럼 실제 예제로 접근해 보죠. 감성분석은 가장 많이 활용되는 텍스트 분류 기법 입니다. 사용자의 댓글이나 리뷰 등을 긍정 또는 부정으로 분류하여 마케팅이나 서비스 향상에 활용하고자 하는 방법 입니다. 물론 실제로 딥러닝 이전에는 Naive Bayes를 통해 접근하기보단, 각 클래스 별 어휘 사전(vocabulary)을 만들어 해당 어휘의 등장 여부에 따라 판단하는 방법을 주로 사용하곤 하였습니다.

$$
\begin{gathered}
\mathcal{C}=\{\color{blue}pos\color{default},\color{red}neg\color{default}\} \\
\mathcal{D}=\{d_1,d_2,\cdots\}
\end{gathered}
$$

위와 같이 긍정($\color{blue}pos\color{default}$)과 부정($\color{red}neg\color{default}$)으로 클래스가 구성($\mathcal{C}$되어 있고, 문서 $d$로 구성된 데이터 $\mathcal{D}$가 있습니다.

이때, 우리에게 "I am happy to see this movie!"라는 문장이 주어졌을 때, 이 문장이 긍정인지 부정인지 판단해 보겠습니다.

$$
\begin{aligned}
P(\color{blue}pos\color{default}|I,am,happy,to,see,this,movie,!)&= \frac{P(I,am,happy,to,see,this,movie,!|\color{blue}pos\color{default})P(\color{blue}pos\color{default})}{P(I,am,happy,to,see,this,movie,!)}\\
&\approx \frac{P(I|\color{blue}pos\color{default})P(am|\color{blue}pos\color{default})P(happy|\color{blue}pos\color{default})\cdots P(!|\color{blue}pos\color{default})P(\color{blue}pos\color{default})}{P(I,am,happy,to,see,this,movie,!)}
\end{aligned}
$$

Naive Bayes의 수식을 활용하여 단어의 조합에 대한 확률을 각각 분해할 수 있습니다. 그리고 그 확률들은 아래와 같이 데이터 $\mathcal{D}$에서의 출현 빈도를 통해 구할 수 있습니다.

$$
\begin{aligned}
P(happy|\color{blue}pos\color{default})&\approx\frac{Count(happy, \color{blue}pos\color{default})}{\sum_{j=1}^{|V|}{Count(w_j,\color{blue}pos\color{default})}} \\
P(\color{blue}pos\color{default})&\approx\frac{Count(\color{blue}pos\color{default})}{|\mathcal{D}|}
\end{aligned}
$$

마찬가지로 부정 감성에 대해 같은 작업을 반복 할 수 있습니다.

$$
\begin{aligned}
P(\color{red}neg\color{default}|I,am,happy,to,see,this,movie,!)&= \frac{P(I,am,happy,to,see,this,movie,!|\color{red}neg\color{default})P(\color{red}neg\color{default})}{P(I,am,happy,to,see,this,movie,!)}\\
&\approx \frac{P(I|\color{red}neg\color{default})P(am|\color{red}neg\color{default})P(happy|\color{red}neg\color{default})\cdots P(!|\color{red}neg\color{default})P(\color{red}neg\color{default})}{P(I,am,happy,to,see,this,movie,!)} \\
\\
P(happy|\color{red}neg\color{default})&\approx\frac{Count(happy, \color{red}neg\color{default})}{\sum_{j=1}^{|V|}{Count(w_j,\color{red}neg\color{default})}} \\
P(\color{red}neg\color{default})&\approx\frac{Count(\color{red}neg\color{default})}{|\mathcal{D}|}
\end{aligned}
$$

## Add-one Smoothing

여기에 문제가 하나 있습니다. 만약 훈련 데이터에서 $Count(happy, \color{red}neg\color{default})$가 $0$이었다면 $P(happy|\color{red}neg\color{default})=0$이 되겠지만, 그저 훈련 데이터에 존재하지 않는 경우라고 해서 실제 출현 확률을 $0$으로 여기는 것은 매우 위험한 일 입니다.

$$
\begin{gathered}
P(happy|\color{red}neg\color{default})\approx\frac{Count(happy, \color{red}neg\color{default})}{\sum_{j=1}^{|V|}{Count(w_j,\color{red}neg\color{default})}}=0, \\
\\
\text{where }Count(happy, \color{red}neg\color{default})=0.
\end{gathered}
$$

따라서 우리는 이런 경우를 위하여 각 출현횟수에 $1$을 더해주어 간단하게 문제를 완화 할 수 있습니다. 물론 완벽한 해결법은 아니지만, Naive Bayes의 가정과 마찬가지로 간단하고 강력합니다.

$$
\tilde{P}(w|c)=\frac{Count(w,c)+1}{\big(\sum_{j=1}^{|V|}{Count(w_j,c)}\big)+|V|}
$$

## Conclusion

위와 같이 Naive Bayes를 통해서 단순히 출현빈도를 세는 것처럼 쉽고 간단하지만 강력하게 감성분석을 구현 할 수 있습니다. 하지만 문장 "I am not happy to see this movie!"라는 문장이 주어지면 어떻게 될까요? "not"이 추가 되었을 뿐이지만 문장의 뜻은 반대가 되었습니다. 

$$
\begin{gathered}
P(\color{blue}pos\color{default}|I,am,not,happy,to,see,this,movie,!) \\
P(\color{red}neg\color{default}|I,am,not,happy,to,see,this,movie,!)
\end{gathered}
$$

"not"은 "happy"를 수식하기 때문에 두 단어를 독립적으로 보는 것은 옳지 않을 수 있습니다. 

$$
P(not,happy) \neq P(not)P(happy)
$$

사실 문장은 단어들이 순서대로 나타나서 의미를 이루기 때문에, 각 단어의 출현 여부도 중요하지만, 각 단어 사이의 순서로 인해 생기는 관계도 무시할 수 없습니다. 하지만 Naive Bayes의 가정은 언어의 이런 특징을 단순화하여 접근하기 때문에 한계가 있습니다.

하지만, 레이블(labeled) 데이터가 매우 적은 경우에는 딥러닝보다 이런 간단한 방법을 사용하는 것이 훨씬 더 나은 대안이 될 수도 있습니다. 이처럼 Naive Bayes는 매우 간단하고 강력하지만, Naive Bayes를 강력하게 만들어준 가정이 가져오는 단점 또한 명확합니다.