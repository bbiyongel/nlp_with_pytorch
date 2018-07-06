# \(Vanilla\) Recurrent Neural Network

기존 신경망은 정해진 입력 $$x$$를 받아 $$y$$를 출력해 주는 형태였습니다.

![](/assets/rnn-fc.png)

$$
y=f(x)
$$

하지만 recurrent neural network \(순환신경망, RNN\)은 입력 $$x_t$$와 직전 자신의 상태\(hidden state\) $$h_{t-1}$$를 참조하여 현재 자신의 상태 $$h_t$$를 결정하는 작업을 여러 time-step에 걸쳐 수행 합니다. 각 time-step별 RNN의 상태는 경우에 따라 출력이 되기도 합니다.

![](/assets/rnn-basic.png)

$$
h_t=f(x_t, h_{t-1})
$$

### Feed-forward

기본적인 RNN을 활용한 feed-forward 계산의 흐름은 아래와 같습니다. 아래의 그림은 각 time-step 별로 입력 $$x_t$$와 이전 time-step의 $$h_t$$가 RNN으로 들어가서 출력으로 $$h_t$$를 반환하는 모습입니다. 이렇게 얻어낸 $$h_t$$들을 $$\hat{y}_t$$로 삼아서 정답인 $$y_t$$와 비교하여 손실(loss) $$\mathcal{L}$$을 계산 합니다.

![](/assets/rnn-basic-architecture.png)

위 그림을 수식으로 표현하면 아래와 같습니다. 함수 $$f$$는 $$x_t$$와 $$h_{t-1}$$을 입력으로 받아서 파라미터 $$\theta$$를 통해 $$h_t$$를 계산 합니다.

$$
\begin{aligned}
\hat{y}_t=h_t&=f(x_t,h_{t-1};\theta) \\
&=\tanh(w_{ih}x_t+b_{ih}+w_{hh}h_{t−1}+b_{hh}) \\
&where~\theta=[w_{ih};b_{ih};w_{hh};b_{hh}].
\end{aligned}
$$

위와 같이 각 time-step별로 $$y_t$$를 계산하여 아래의 수식처럼 모든 time-step에 대한 손실(loss) $$\mathcal{L}$$을 구합니다.

$$
\mathcal{L}=\frac{1}{n}\sum_{t=1}^{n}{loss(y_t,\hat{y}_t)}
$$

## Back-propagation Through Time (BPTT)

그럼 이렇게 feed-forward 된 이후에 오류의 back-propagation(역전파)은 어떻게 될까요? 앞서 구한 손실 $$\mathcal{L}$$에 미분을 통해 back-propagation 하게 되면, 각 time-step 별로 뒤($$t$$가 큰 time-step)로부터 gradient가 구해지고, 이전 time-step ($$t-1$$)의 gradient에 더해지게 됩니다. 즉, $$t$$가 $$0$$에 가까워질수록 RNN 파라미터($$\theta$$)의 gradient는 각 time-step 별 gradient가 더해져 점점 커지게 됩니다.

![](/assets/rnn-back-prop.png)

위 그림에서는 붉은색이 점점 짙어지는 것으로 그런 RNN back-propagation의 속성을 나타내었습니다. 이 속성을 back-propagation through time(BPTT)이라고 합니다.

## Multi-layer RNN

![](/assets/rnn-multi-layer.png)

## Bi-directional RNN

![](/assets/rnn-bidirectional.png)

## How to Apply to NLP

### Use only last hidden state as output

![](/assets/rnn-apply-1.png)

$$
\text{softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}
$$

### Use all hidden states as output

![](/assets/rnn-apply-2.png)

## Gradient vanishing & exploding

## 코드

## 설명



