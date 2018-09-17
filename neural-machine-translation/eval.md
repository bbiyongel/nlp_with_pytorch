# Evaluation

번역기의 성능을 평가하는 방법은 크게 두 가지로 나눌 수 있습니다. 정성적(implicit) 평가와 정량적(explicit) 평가 방식입니다.

## Implicit Evaluation

정성평가 방식은 보통 사람이 번역된 문장을 채점하는 형태로 이루어집니다. 사람은 선입견 등이 채점하는데 있어서 방해요소로 작용될 수 있기 때문에, 보통은 blind test를 통해서 채점합니다. 이를 위해서 여러개의 다른 알고리즘을 통해 (또는 경쟁사의) 여러 번역결과를 누구의 것인지 밝히지 않은 채, 채점하여 우열을 가립니다. 이 방식은 가장 정확하다고 할 수 있지만, 자원과 시간이 많이 드는 단점이 있습니다.

## Explicit Evalution

위의 단점 때문에, 보통은 자동화 된 정량평가를 주로 수행합니다. 두 평가를 모두 주기적으로 수행하되, 정성평가의 평가 주기를 좀 더 길게 가져가거나, 무언가 확실한 성능의 jump가 이루어졌을 때 수행하는 편 입니다.

### Cross Entropy and Perplexity

신경망 기계번역도 기본적으로 매 time-step마다 최고 확률을 갖는 단어를 선택(분류) 하는 작업이기 때문에 기본적으로 분류(classification)작업에 속합니다. 따라서 Cross Entropy를 손실함수(loss function)로 사용합니다. 이전 섹션에서 언급하였듯이, 신경망 기계번역도 조건부 언어모델이기 때문에 perplexity를 통해 성능을 측정할 수 있습니다. 그러므로 이전 언어모델 챕터에서 다루었듯이, cross entropy loss 값에 exponantial을 취하여 perplexity(PPL) 값을 얻을 수 있습니다.

### BLEU

위의 PPL은 우리가 사용하는 Loss function과 직결되어 바로 알 수 있는 간편함이 있지만, 사실 안타깝게도 실제 번역기의 성능과 완벽한 비례관계에 있다고 할 수는 없습니다. Cross Entropy의 수식을 해석 해 보면, 각 time-step 별 실제 정답에 해당하는 단어의 확률만 채점하기 때문입니다.

|원문|I|love|to|go|to|school|.|
|-|-|-|-|-|-|-|-|
|index|0|1|2|3|4|5|6|
|정답|나는|학교에|가는|것을|좋아한다|.| |
|번역1|학교에|가는|것을|좋아한다|나는|.| |
|번역2|나는|오락실에|가는|것을|싫어한다|.| |

예를 들어, 번역1은 cross entropy loss에 의하면 매우 높은 loss값을 가집니다. 하지만 번역2는 번역1에 비해 완전 틀린 번역이지만 loss가 훨씬 낮을 겁니다. 따라서 실제 번역문의 품질과 cross entropy 사이에는 (특히 teacher forcing 방식이 더해져) 괴리가 있습니다. 이러한 간극을 줄이기 위해 여러가지 방법들이 제시되었습니다 -- [METEOR](https://en.wikipedia.org/wiki/METEOR), [BLEU](https://en.wikipedia.org/wiki/BLEU). 이번 섹션은 그 중 가장 널리 쓰이는 BLEU에 대해 짚고 넘어가겠습니다.

$$
\begin{gathered}
BLEU=\text{brevity-penalty}*\prod_{n=1}^{N}{p_n^{w_n}} \\
\text{where brevity penalty}=\min(1, \frac{|\text{prediction}|}{|\text{reference}|}) \\
\text{and }p_n\text{ is precision of }n\text{-gram and }w_n\text{ is weight that }w_n=\frac{1}{2^n}.
\end{gathered}
$$

BLEU는 정답 문장과 예측 문장 사이에 일치하는 n-gram의 갯수의 비율의 기하평균에 따라 점수가 매겨집니다. brevity penalty는 예측 된 번역문이 정답 문장보다 짧을 경우 점수가 좋아지는 것을 방지하기 위함입니다. 보통 위 수식의 결과 값에 100을 곱하여 0-100의 scale로 점수를 표현합니다. 실제 위의 예제 '번역1'에서 나타난 2-gram을 count하여 간단하게 BLEU를 측정 하여 보겠습니다.

|2-gram|count|hit count|
|-|-|-|
|BOS 학교에|1|0|
|학교에 가는|1|1|
|가는 것을|1|1|
|것을 좋아한다|1|1|
|좋아한다 나는|1|0|
|나는 .|1|0|
|. EOS|1|1|
|합계|7|4|

따라서 2-gram의 BLEU score는 ${4}/{7}$이 됩니다. 이번에는 '번역2'에 대한 2-gram BLEU를 측정 해 보겠습니다.

|2-gram|count|hit count|
|-|-|-|
|BOS 나는|1|1|
|나는 오락실에|1|0|
|오락실에 가는|1|0|
|가는 것을|1|1|
|것을 싫어한다|1|0|
|싫어한다 .|1|0|
|. EOS|1|1|
|합계|7|3|

'번역2'의 2-gram BLEU는 ${3}/{7}$가 나왔습니다. 그러므로 (brevity penalty나 1-gram, 2-gram, 3-gram, 4-gram의 점수를 평균내지는 않았지만) 2-gram에 한해서 ${4}/{7}>{3}/{7}$이므로 '번역1'이 더 잘 번역되었다고 볼 수 있습니다. 즉, 위의 예제에서 '번역1'이 '번역2'보다 일치하는 n-gram이 더 많으므로 더 높은 BLEU 점수를 획득할 수 있습니다. 이와 같이 BLEU는 대체로 실제 정성평가의 결과와 일치하는 경향이 있다고 여겨집니다.

결론적으로 우리는 성능 평가 결과를 해석할 때 Perplexity(=Loss)는 낮을수록 좋고, BLEU는 높을수록 좋다고 합니다. 앞으로 설명할 알고리즘들의 성능을 평가할 때 참고 바랍니다. 실제 성능을 측정하기 위해서는 보통 SMT 프레임웍인 MOSES의 [multi-bleu.perl](https://github.com/google/seq2seq/blob/master/bin/tools/multi-bleu.perl)을 주로 사용합니다.
