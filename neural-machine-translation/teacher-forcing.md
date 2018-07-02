# Auto-regressive and Teacher Focing

많은 분들이 여기까지 잘 따라왔다면 궁금즘을 하나 가질 수 있습니다. Decoder의 입력으로 이전 time-step의 출력이 들어가는것이 훈련 때도 같은 것인가? 사실, 안타깝게도 seq2seq의 기본적인 훈련 방식은 추론(inference)할 때의 방식과 상이합니다.

## Auto-regressive

## Teacher Forcing

중요한 점은 **훈련(training)시에는 decoder의 입력으로 이전 time-step의 decoder의 출력값이 아닌, 실제 $$ Y $$가 들어간다**는 것입니다. 하지만, 추론(inference) 할 때에는 실제 $$ Y $$를 모르기 때문에, 이전 time-step에서 계산되어 나온 $$ \hat{y_{t-1}} $$를 decoder의 입력으로 사용합니다. 이 훈련 방법을 ***Teacher Forcing***이라고 합니다. Teacher Forcing이 필요한 이유는 NMT의 수식을 살펴보면 알 수 있습니다. 해당 time-step의 단어를 구할 때 수식은 아래와 같습니다.

$$
\hat{y}_t=argmax{P(y_t|X,y_{<t};\theta)}~where~X=\{x_1,x_2,\cdots,x_n\}
$$

위와 같이 조건부에 $$ \hat{y}_{<t} $$가 들어가는 것이 아닌, $$ y_{<t} $$가 들어가는 것이기 때문에, 훈련시에는 이전 time-step의 출력을 넣어줄 수 없습니다. 만약 넣어주게 된다면 해당 time-step의 decoder에겐 잘못된 것을 가르쳐 주는 꼴이 될 것입니다. 따라서 training 할 때에는 모든 time-step을 한번에 계산할 수 있습니다. 그러므로 decoder도 각 time-step별이 아닌 한번에 수식을 정리할 수 있습니다.

$$
H^{tgt}=RNN_{dec}(emb_{tgt}([BOS;Y[:-1]]),h_{n}^{src})
$$

