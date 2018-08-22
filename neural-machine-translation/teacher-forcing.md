# Auto-regressive and Teacher Focing

많은 분들이 여기까지 잘 따라왔다면 궁금즘을 하나 가질 수 있습니다. Decoder의 입력으로 이전 time-step의 출력이 들어가는것이 훈련 때도 같은 것인가? 사실, 안타깝게도 seq2seq의 기본적인 훈련(training) 방식은 추론(inference)할 때의 방식과 상이합니다.

## Auto-regressive

Sequence-to-sequence의 훈련 방식과 추론 방식의 차이는 근본적으로 auto-regressive라는 속성 때문에 생겨납니다. Auto-regressive는 과거의 자신의 값을 참조하여 현재의 값을 추론(또는 예측) 해 내기 때문에 붙은 이름입니다. 이는 수식에서도 확인 할 수 있습니다. 아래는 전체적인 신경망 기계번역의 수식 입니다.

$$
\begin{aligned}
Y&=argmax_{Y}P(Y|X)=argmax_{Y}\prod_{i=1}^{n}{P(y_i|X,y_{<i})} \\
or \\
y_i&=argmax_{y}P(y|X,y_{<i}) \\
&where~y_0=BOS.
\end{aligned}
$$

위와 같이 현재 time-step의 출력값 $y_t$는 encoder의 입력 문장(또는 시퀀스) $X$와 이전 time-step까지의 $y_{<t}$를 조건부로 받아 결정 되기 때문에, 과거 자신의 값을 참조하게 되는 것 입니다. 

이것은 과거에 잘못된 예측을 하게 되면 점점 시간이 지날수록 더 큰 잘못된 예측을 할 가능성을 야기하기도 합니다. 또한, 과거의 결과값에 따라 문장(또는 시퀀스)의 구성이 바뀔 뿐만 아니라, 그 길이 마저도 바뀌게 됩니다. 따라서 우리는 이런 auto-regressive 속성을 유지한 채 훈련을 할 수 없습니다.

$$
\hat{y}_t=argmax_{y}{P(y|X,y_{<t};\theta)}~where~X=\{x_1,x_2,\cdots,x_n\}~and~Y=\{y_0,y_1,\cdots,y_{m+1}\}
$$

훈련 할 때에 각 time-step 별 수식은 다음과 같습니다. 위와 같이 조건부에 $\hat{y}_{<t}$가 들어가는 것이 아닌, $y_{<t}$가 들어가는 것이기 때문에, 훈련시에는 이전 time-step의 출력 $\hat{y}_{<t}$을 현재 time-step의 입력으로 넣어줄 수 없습니다. 만약 넣어주게 된다면 현재 time-step의 decoder에겐 잘못된 것을 가르쳐 주는 꼴이 될 것입니다.

$$
\begin{aligned}
\mathcal{L}(Y)&=-\sum_{i=1}^{m+1}{\log{P(y_i|X,y_{<i};\theta)}} \\
\theta &\leftarrow \theta - \lambda\frac{1}{N}\sum_{i=1}^{N}{\mathcal{L}(Y_i)}
\end{aligned}
$$

또한, 실제 손실함수(loss function)을 계산하여 gradient descent를 수행할 때도, $\hat{y_i}$의 확률을 사용하지 않고, softmax layer에서 정답에 해당하는 $y_i$의 인덱스(index)에 있는 로그(log)확률값을 사용 합니다.

## Teacher Forcing

따라서 우리는 Teacher Forcing이라고 불리는 방법을 사용하여 훈련 합니다.

중요한 점은 **훈련(training)시에는 decoder의 입력으로 이전 time-step의 decoder의 출력값이 아닌, 실제 $Y$가 들어간다**는 것입니다. 하지만, 추론(inference) 할 때에는 실제 $Y$를 모르기 때문에, 이전 time-step에서 계산되어 나온 $\hat{y_{t-1}}$를 decoder의 입력으로 사용합니다. 이 훈련 방법을 ***Teacher Forcing***이라고 합니다.

추론(inference) 할 때에는 auto-regressive 속성 때문에 과거 자신을 참조해야 합니다. 따라서 각 time-step 별로 순차적(sequential)으로 진행해야 합니다. 하지만 훈련(training) 할 때에는 입력값이 정해져 있으므로, 모든 time-step을 한번에 계산할 수 있습니다. 그러므로 decoder도 모든 time-step을 합쳐 수식을 정리할 수 있습니다.

$$
H^{tgt}=RNN_{dec}(emb_{tgt}([BOS;Y[:-1]]),h_{n}^{src})
$$

이런 auto-regressive 속성 및 teacher forcing 방법은 신경망 언어모델(NNLM)에도 똑같이 적용되는 문제 입니다. 하지만 언어모델의 경우에는 perplexity는 문장의 확률과 직접적으로 연관이 있기 때문에, 큰 문제가 되지 않는 반면에 기계번역에서는 좀 더 큰 문제로 다가옵니다. 이에 대해서는 추후 다루도록 하겠습니다.