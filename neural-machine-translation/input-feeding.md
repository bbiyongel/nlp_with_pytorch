# Input Feeding

Decoder output과 Attention 결과값을 concatenate한 이후에 Generator 모듈에서 softmax를 취하여 $$ \hat{y}_{t+1} $$을 구합니다. 하지만 이러한 softmax 과정에서 많은 정보(예를 들어 attention 정보 등)가 손실됩니다. 따라서 단순히 다음 time-step에 $$ \hat{y}_{t+1} $$을 feeding 하는 것보다, concatenation layer의 출력도 같이 feeding 해주면 정보의 손실 없이 더 좋은 효과를 얻을 수 있습니다.

![](/assets/seq2seq_with_attention_and_input_feeding.png)

## 1. 단점

## 2. 성능 실험

![https://arxiv.org/pdf/1508.04025.pdf](/assets/attention_evalution_result.png)  
WMT’14 English-German results \[Loung, arXiv 2015\]

현재 방식을 처음 제안한 [\[Loung et al.2015\] Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)에서는 실험 결과를 위와 같이 주장하였습니다. 실험 대상은 아래와 같습니다.

* Baseline: 기본적인 seq2seq 모델
* Reverse: Bi-directional LSTM을 encoder에 적용
* Dropout: probability 0.2
* Global Attention
* Input Feeding

우리는 이 실험에서 attention과 input feeding을 사용함으로써, 훨씬 더 나은 성능을 얻을 수 있음을 알 수 있습니다.

