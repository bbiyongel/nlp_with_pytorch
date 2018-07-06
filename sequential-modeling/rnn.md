# \(Vanilla\) Recurrent Neural Network

기존 신경망은 정해진 입력 $$x$$를 받아 $$y$$를 출력해 주는 형태였습니다.

![](/assets/rnn-fc.png)
$$
y=f(x)
$$


하지만 recurrent neural network \(순환신경망, RNN\)은 입력과 직전 출력값을 참조하여 현재 출력값을 반환하는 작업을 여러 time-step에 걸쳐 수행 합니다.

![](/assets/rnn-basic.png)


$$
h_t=f(x_t, h_{t-1})
$$


### Feed-forward

### Back-propagation

## Gradient vanishing & exploding

## 코드

## 설명



