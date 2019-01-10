# Auto-regressive 속성과 Teacher Forcing 훈련 방법

많은 분들이 여기까지 잘 따라왔다면 궁금즘을 하나 가질 수 있습니다. 디코더의 입력으로 이전 time-step의 출력이 들어가는것이 훈련 때도 같은 것인가? 사실, 안타깝게도 sequence-to-sequence의 기본적인 훈련(training) 방식은 추론(inference)할 때의 방식과 상이합니다.

## Auto-regressive 속성

Sequence-to-sequence의 훈련 방식과 추론 방식의 차이는 근본적으로 auto-regressive라는 속성 때문에 생겨납니다. Auto-regressive는 과거의 자신의 값을 참조하여 현재의 값을 추론(또는 예측)하는 특성을 가리키는 이름 입니다. 이는 수식에서도 확인 할 수 있습니다. 예를 들어 아래는 전체적인 신경망 기계번역의 수식 입니다.

$$\begin{gathered}
\hat{Y}=\underset{Y\in\mathcal{Y}}{\text{argmax}}P(Y|X)=\underset{Y\in\mathcal{Y}}{\text{argmax}}\prod_{i=1}^{n}{P(\text{y}_i|X,\hat{y}_{<i})} \\
\text{or} \\
\hat{y}_t=\underset{y\in\mathcal{Y}}{\text{argmax }}{P(\text{y}_t|X,\hat{y}_{<t};\theta)} \\
\text{where }X=\{x_1,x_2,\cdots,x_n\}\text{, }Y=\{y_0,y_1,\cdots,y_{m+1}\}\text{ and }\text{where }y_0=\text{BOS}.
\end{gathered}$$

위와 같이 현재 time-step의 출력값 $y_t$ 는 인코더의 입력 문장(또는 시퀀스) $X$ 와 이전 time-step까지의 $y_{<t}$ 를 조건부로 받아 결정 되기 때문에, 과거 자신의 값을 참조하게 되는 것 입니다. 이러한 점은 과거에 잘못된 예측을 하게 되면 점점 시간이 지날수록 더 큰 잘못된 예측을 할 가능성을 야기하기도 합니다. 또한, 과거의 결과값에 따라 문장(또는 시퀀스)의 구성이 바뀔 뿐만 아니라, 예측 문장(시퀀스)의 길이 마저도 바뀌게 됩니다. 학습 과정에서는 이미 정답을 알고 있고, 현재 모델의 예측값과 정답과의 차이를 통해 학습하기 때문에, 우리는 auto-regressive 속성을 유지한 채 훈련을 할 수 없습니다.

## Teacher Forcing 훈련 방법

따라서 우리는 Teacher Forcing이라고 불리는 방법을 사용하여 훈련 합니다. 훈련 할 때에 각 time-step 별 수식은 아래와 같습니다. 위와 같이 조건부에 $\hat{y}_{<t}$ 가 들어가는 것이 아닌, $y_{<t}$ 가 들어가는 것이기 때문에, 훈련시에는 이전 time-step의 출력 $\hat{y}_{<t}$ 을 현재 time-step의 입력으로 넣어줄 수 없습니다.

$$\begin{gathered}
\mathcal{L}(Y)=-\sum_{i=1}^{m+1}{\log{P(\text{y}_i|X,y_{<i};\theta)}} \\
\theta\leftarrow\theta-\lambda\frac{1}{N}\sum_{i=1}^{N}{\mathcal{L}(Y_i)} \\
\text{where }(X,Y)\sim\mathcal{B}=\{X_i,Y_i\}_{i=1}^N
\end{gathered}$$

또한, 실제 손실함수(loss function)을 계산하여 그래디언트 디센트를 수행할 때도, softmax를 통해 얻은 확률 분포 $\log{P(\text{y}_i|X,y_{<i};\theta)}$ 에서 해당 time-step의 $\text{argmax}$ 값인 $\hat{y_i}$ 의 확률을 사용하지 않고, 크로스엔트로피의 수식에 따라서 정답에 해당하는 $y_i$ 의 인덱스(index)에 있는 로그(log)확률값 $\log{P(\text{y}_i=y_i|X,y_{<i};\theta)}$ 을 사용 합니다.

중요한 점은 훈련시에는 디코더의 입력으로 이전 time-step의 디코더의 출력값이 아닌, 정답 $Y$ 가 들어간다는 것입니다. 하지만 추론(inference) 할 때에는 정답 $Y$ 를 모르기 때문에, 이전 time-step에서 계산되어 나온 $\hat{y}_{t-1}$ 를 디코더의 입력으로 사용합니다. 이렇게 추론과 상이한 입력을 넣어주는 훈련 방법을 Teacher Forcing이라고 합니다.

이전에 언급하였듯이, 추론 할 때에는 auto-regressive 속성 때문에 과거 자기자신을 참조해야 합니다. 따라서 이전 time-step의 자기자신의 상태를 알기 위해서, 각 time-step 별로 순차적(sequential)으로 진행해야 합니다. 하지만 훈련 할 때에는 입력값이 정해져 있으므로, 모든 time-step을 한번에 계산할 수 있습니다. 그러므로 input feeding이 존재하지 않는 디코더는 모든 time-step을 합쳐 수식을 정리할 수 있습니다.

$$H^{tgt}=\text{RNN}_{dec}(\text{emb}_{tgt}([BOS;Y[:-1]]),h_{n}^{src})$$

하지만 바로 앞 섹션에서 다루엇듯이, 디코더의 input feeding은 이전 time-step의 softmax 이전 레이어의 값을 단어 임베딩 벡터와 함께 받아야 하기 때문에 위와 같이 모든 time-step을 한번에 계산하는 것은 불가능 합니다. 따라서 input feeding이 추가 된 디코더의 수식은 아래와 같이 정의 됩니다.

$$h_{t}^{tgt}=\text{RNN}_{dec}([\text{emb}_{tgt}(y_{t-1});\tilde{h}_{t-1}^{tgt}], h_{t-1}^{tgt})\text{ where }h_{0}^{tgt}=h_{n}^{src}\text{ and }y_{0}=\text{BOS}.$$

이런 auto-regressive 속성 및 teacher forcing 방법은 신경망 언어모델(NNLM)에도 똑같이 적용되는 문제 입니다. 하지만 언어모델의 경우에는 perplexity는 문장의 확률과 직접적으로 연관이 있기 때문에, 큰 문제가 되지 않는 반면에 기계번역에서는 좀 더 큰 문제로 다가옵니다. 이에 대해서는 이어지는 섹션에서 다루도록 하겠습니다.
