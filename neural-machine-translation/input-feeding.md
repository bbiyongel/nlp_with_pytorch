# Input Feeding

Decoder output과 Attention 결과값을 concatenate한 이후에 Generator 모듈에서 softmax를 취하여 $$ \hat{y}_{t} $$을 구합니다. 하지만 이러한 softmax 과정에서 많은 정보(예를 들어 attention 정보 등)가 손실됩니다. 따라서 단순히 다음 time-step에 $$ \hat{y}_{t} $$을 feeding 하는 것보다, concatenation layer의 출력도 같이 feeding 해주면 정보의 손실 없이 더 좋은 효과를 얻을 수 있습니다.

![](/assets/seq2seq_with_attention_and_input_feeding.png)

$$ y $$와 달리 concatenation layer의 출력은 $$ y $$가 embedding layer에서 dense vector(=embedding vector)로 변환되고 난 이후에 embedding vector와 concatenate되어 decoder RNN에 입력으로 주어지게 됩니다. 이러한 과정을 ***input feeding***이라고 합니다.

$$
h_{t}^{src} = RNN(emb_{src}(x_t), h_{t-1}^{src})
$$
$$
H^{src} = [h_{1}^{src}; h_{2}^{src}; \cdots; h_{n}^{src}]
$$
$$
h_{t}^{tgt} = RNN([emb_{tgt}(y_{t-1});\tilde{h}_{t-1}^{tgt}], h_{t-1}^{tgt})~~where~h_{0}^{tgt} = h_{n}^{src} and ~y_{0}=BOS
$$
$$
w = softmax({h_{t}^{tgt}}^T W \cdot H^{src})
$$
$$
c = H^{src} \cdot w~~~~~and~c~is~a~context~vector
$$
$$
\tilde{h}_{t}^{tgt}=\tanh([h_{t}^{tgt}; c])
$$
$$
\hat{y}_{t}=softmax(\tilde{h}_{t}^{tgt})
$$

## 1. 단점

이 방식은 ***훈련 속도 저하***라는 단점을 가집니다. input feeding이전 방식에서는 훈련 할 때에는 모든 $$ Y $$를 알고 있기 때문에, encoder와 마찬가지로 decoder도 모든 time-step에 대해서 한번에 ***feed-forward*** 작업이 가능했습니다. 하지만 input feeding으로 인해, decoder RNN의 input으로 이전 time-step의 결과가 필요하게 되어, decoder ***feed-forward*** 할 때에 time-step 별로 sequential하게 계산을 해야 합니다.

하지만 이 단점이 크게 부각되지 않는 이유는 어차피 ***inference*** 단계에서는 decoder는 input feeding이 아니더라도 time-step 별로 sequential하게 계산되어야 하기 때문입니다. inference 단계에서는 이전 time-step의 output인 $$ \hat{y}_t $$를 decoder(정확하게는 decoder 이전의 embedding layer)의 입력으로 사용해야 하기 때문에, 어쩔 수 없이 병렬처리가 아닌 sequential 하게 계산해야 합니다. 따라서 input feeding으로 인한 속도 저하는 거의 없습니다.

## 2. 성능 실험

|NMT system|Perplexity|BLEU|
|-|-|-|
|Base|10.6|11.3|
|Base + reverse|9.9|12.6(+1.3)|
|Base + reverse + dropout|8.1|14.0(+1.4)|
|Base + reverse + dropout + attention|7.3|16.8(+2.8)|
|Base + reverse + dropout + attention + feed input|6.4|18.1(+1.3)|

WMT’14 English-German results Perplexity(PPL) and BLEU \[Loung, arXiv 2015\]

현재 방식을 처음 제안한 [\[Loung et al.2015\] Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)에서는 실험 결과를 위와 같이 주장하였습니다. 실험 대상은 아래와 같습니다.

* Baseline: 기본적인 seq2seq 모델
* Reverse: Bi-directional LSTM을 encoder에 적용
* Dropout: probability 0.2
* Global Attention
* Input Feeding

우리는 이 실험에서 attention과 input feeding을 사용함으로써, 훨씬 더 나은 성능을 얻을 수 있음을 알 수 있습니다.

