# Input Feeding

Decoder output과 Attention 결과값을 concatenate한 이후에 Generator 모듈에서 softmax를 취하여 $$ y^{\hat}_{t+1} $$을 

![](/assets/seq2seq_with_attention_and_input_feeding.png)

## 성능 실험

![https://arxiv.org/pdf/1508.04025.pdf](/assets/attention_evalution_result.png)  
WMT’14 English-German results \[Loung, arXiv 2015\]

현재 방식을 처음 제안한 [\[Loung et al.2015\] Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)에서는 실험 결과를 위와 같이 주장하였습니다. 실험 대상은 아래와 같습니다.

* Baseline: 기본적인 seq2seq 모델
* Reverse: Bi-directional LSTM을 encoder에 적용
* Dropout: probability 0.2
* Global Attention
* Input Feeding

우리는 이 실험에서 attention과 input feeding을 사용함으로써, 훨씬 더 나은 성능을 얻을 수 있음을 알 수 있습니다.

