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

위의 수식에서 나타나듯이 RNN에서는 ReLU나 다른 활성함수(activation function)을 사용하기보단 $$\tanh$$를 주로 사용합니다.

최종적으로 각 time-step별로 $$y_t$$를 계산하여 아래의 수식처럼 모든 time-step에 대한 손실(loss) $$\mathcal{L}$$을 구합니다.

$$
\mathcal{L}=\frac{1}{n}\sum_{t=1}^{n}{loss(y_t,\hat{y}_t)}
$$

## Back-propagation Through Time (BPTT)

그럼 이렇게 feed-forward 된 이후에 오류의 back-propagation(역전파)은 어떻게 될까요? 앞서 구한 손실 $$\mathcal{L}$$에 미분을 통해 back-propagation 하게 되면, 각 time-step 별로 뒤($$t$$가 큰 time-step)로부터 gradient가 구해지고, 이전 time-step ($$t-1$$)의 gradient에 더해지게 됩니다. 즉, $$t$$가 $$0$$에 가까워질수록 RNN 파라미터($$\theta$$)의 gradient는 각 time-step 별 gradient가 더해져 점점 커지게 됩니다.

![](/assets/rnn-back-prop.png)

위 그림에서는 붉은색이 점점 짙어지는 것으로 그런 RNN back-propagation의 속성을 나타내었습니다. 이 속성을 back-propagation through time(BPTT)이라고 합니다.

이런 RNN back-propagation의 속성으로 인해, 마치 RNN은 time-step의 수 만큼 layer(계층)이 있는 것이나 마찬가지가 됩니다. 따라서 time-step이 길어짐에 따라, 매우 깊은 신경망과 같이 동작 합니다.

## Gradient Vanishing & Exploding

## Multi-layer RNN

기본적으로 Time-step별로 RNN이 동작하지만, 아래의 그림과 같이 한 time-step 내에서 RNN을 여러 층을 쌓아올릴 수 있습니다. 그림상으로 시간의 흐름은 왼쪽에서 오른쪽으로 간다면, 여러 layer를 아래에서 위로 쌓아 올릴 수 있습니다. 당연히 각 층 별로 파라미터 $$\theta$$를 공유하지 않고 따로 갖습니다.

![](/assets/rnn-multi-layer.png)

## Bi-directional RNN

여러 층을 쌓는 방법에 대해 이야기 했다면, 이제 RNN의 방향에 대해서 이야기 할 차례 입니다. 이제까지 다룬 RNN은 $$t$$가 $$1$$에서부터 마지막 time-step 까지 차례로 입력을 받아 진행 하였습니다. 하지만, bi-directional(양방향) RNN을 사용하게 되면, 기존의 정방향과 추가적으로 마지막 time-step에서부터 거꾸로 역방향으로 입력을 받아 진행 합니다. Bi-directional RNN의 경우에도 당연히 정방향과 역방향의 파라미터 $$\theta$$는 공유되지 않습니다.

![](/assets/rnn-bidirectional.png)

보통은 여러 층의 bi-directional RNN을 쌓게 되면, 각 층마다 두 방향의 각 time-step 별 출력(hidden state)값을 이어붙여(concatenate) 다음 층(layer)의 각 방향 별 입력으로 사용하게 됩니다.

## How to Apply to NLP

그럼 위에서 다룬 내용을 바탕으로 RNN을 NLP를 비롯한 실무에서는 어떻게 적용하는지 알아보도록 하겠습니다. 여기서는 RNN을 한개 층만 쌓아 정방향으로만 다룬 것 처럼 묘사하였지만, 여러 층을 양방향으로 쌓아 사용하는 것도 대부분의 경우 가능 합니다.

### Use only last hidden state as output

![](/assets/rnn-apply-1.png)

$$
\text{softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}
$$

### Use all hidden states as output

![](/assets/rnn-apply-2.png)

Bi-directional RNN을 쓸 수 없는 경우에 대한 설명

## Implementation

### RNN

### RNNCell



