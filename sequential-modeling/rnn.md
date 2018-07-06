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

![](/assets/rnn-architecture.png)

$$
\begin{aligned}
\hat{y}_t=h_t&=f(x_t;\theta) \\
&=\tanh(w_{ih}x_t+b_{ih}+w_{hh}h_{t−1}+b_{hh}) \\
&where~\theta=[w_{ih};b_{ih};w_{hh};b_{hh}]. \\
\\
\mathcal{L}&=\frac{1}{n}\sum_{t=1}^{n}{loss(y_t,\hat{y}_t)}
\end{aligned}
$$

$$
\text{softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}
$$

## Back-propagation

## Multi-layer RNN

## Bi-directional RNN

## How to Apply to NLP

## Gradient vanishing & exploding

## 코드

## 설명



