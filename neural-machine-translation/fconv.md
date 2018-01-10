# Fully Convolutional Seq2seq

Neural Machine Translation의 최강자는 Google이라고 모두가 여기고 있을 때, Facebook이 과감하게 이 논문[\[Gehring at el.2017\]](https://arxiv.org/pdf/1705.03122.pdf)을 들고 도전장을 내밀었습니다. RNN방식의 seq2seq 대신에 오직 convolutional layer만을 이용한 방식의 seq2seq를 들고 나와, 기존의 방식에 대비해서 성능과 속도 두마리 토끼를 모두 잡았다고 주장하였습니다.

## 1. Architecture

![](/assets/nmt-fconv-overview.png)

### a. Position Embedding

### b. Attention

