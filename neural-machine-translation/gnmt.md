# Google Neural Machine Translation \(GNMT\)

Google은 2016년 논문([\[Wo at el.2016\]](https://arxiv.org/pdf/1609.08144.pdf)
)을 발표하여 그들의 번역시스템에 대해서 상세히 소개하였습니다. 실제 시스템에 적용된 모델 architecture부터 훈련 algorithm 까지 상세히 기술하였기 때문에, 실제 번역 시스템을 구성하고자 할 때에 좋은 reference가 될 수 있습니다. 또한 다른 논문들에서 실험 결과에 대해 설명할 때, GNMT를 upper boundary baseline으로 참조하기도 합니다.

## 1. Model Architecture

Google도 seq2seq 기반의 모델을 구성하였습니다. 다만, 구글은 훨씬 방대한 데이터셋을 가지고 있기 때문에 그에 맞는 깊은 모델을 구성하였습니다. 따라서 아래에 소개될 방법들이 깊은 모델들을 효율적으로 훈련 할 수 있도록 사용되었습니다.

### a. Residual Connection

![](/assets/nmt-gnmt-1.png)

보통 LSTM layer를 4개 이상 쌓기 시작하면 모델이 deeper해 짐에 따라서 성능 효율이 저하되기 시작합니다. 따라서 Google은 깊은 모델은 효율적으로 훈련시키기 위하여 residual connection을 적용하였습니다.

### b. Bi-directional Encoder for First Layer

![](/assets/nmt-gnmt-2.png)

또한, 모든 LSTM stack에 대해서 bi-directional LSTM을 적용하는 대신에, 첫번째 층에 대해서만 bi-directional LSTM을 적용하였습니다. 따라서 training 및 inference 속도에 개선이 있었습니다.

## 2. Segmentation Approachs

### a. Wordpiece Model

구글도 마찬가지로 BPE 모델을 사용하여 tokenization을 수행하였습니다. 그리고 그들은 그들의 tokenizer를 오픈소스로 공개하였습니다. -- [SentencePiece: https://github.com/google/sentencepiece](https://github.com/google/sentencepiece) 마찬가지로 아래와 같이 띄어쓰기는 underscore로 치환하고, 단어를 subword별로 통계에 따라 segmentation 합니다.

- original: Jet makers feud over seat width with big orders at stake
- wordpieces: _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake

## 3. Training Criteria

![](/assets/nmt-gnmt-5.png)

Google은 후에 설명할 Reinforcement Learning 기법을 사용하여 Maximum Likelihood Estimation (MLE)방식의 훈련된 모델에 fine-tuning을 수행하였습니다. 따라서 위의 테이블과 같은 추가적이 성능 개선을 얻어낼 수 있었습니다. 이러한 RL 기법은 다음 챕터에서 소개하도록 하겠습니다.

## 4. Quantization

실제 Neural Network을 사용한 product를 개발할 때에는 여러가지 어려움에 부딪히게 됩니다. 이때, Quantization을 도입함으로써 아래와 같은 여러가지 이점을 얻을 수 있습니다.

- 계산량을 줄여 자원의 효율적 사용과 응답시간의 감소를 얻을 수 있다.
- 모델의 실제 저장되는 크기를 줄여 deploy를 효율적으로 할 수 있다.
- 부가적으로 regularization의 효과를 볼 수 있다. (아래 테이블 참조)

![](/assets/nmt-gnmt-3.png)

## 5. Search

### a. Length Penalty and Coverage Penalty

Google은 기존에 소개한 ***Length Penalty***에 추가로 ***Coverage Penalty***를 사용하여 좀 더 성능을 끌어올렸습니다. Coverage penalty는 attention weight(probability)의 값의 분포에 따라서 매겨집니다. 이 penalty는 좀 더 attention이 고루 잘 퍼지게 하기 위함입니다.

$$
s(Y, X) = \log{P(Y|X)}/lp(Y) + cp(X; Y)
$$
$$
lp(Y) = \frac{(5+|Y|)^\alpha}{(5+1)^\alpha}
$$
$$
cp(X; Y) = \beta * \sum_{i=1}^{|X|}{\log{(\min{(\sum_{j=1}^{|Y|}{p_{i,j}}, 1.0)})}}
$$
$$
where~p_{i,j}~is~the~attention~weight~of~the~j\text{-}th~target~word~y_j~on~the~i\text{-}th~source~word~x_i.
$$

Coverage penalty의 수식을 들여다보면, 각 source word $$ x_i $$별로 attention weight의 합을 구하고, 그것의 평균(=합)을 내는 것을 볼 수 있습니다. ***log***를 취했기 때문에 그 중에 attention weight가 편중되어 있다면, 편중되지 않은 source word는 매우 작은 음수 값을 가질 것이기 때문에 좋은 점수를 받을 수 없을 겁니다.

## 6. Training Procedure

![](/assets/nmt-gnmt-4.png)

Google은 stochastic gradient descent (SGD)를 써서 훈련 시키는 것 보다, Adam과 섞어 사용하면 (epoch 1까지 Adam) 더 좋은 성능을 발휘하는 것을 확인하였습니다.

## 7. Evaluation

![](/assets/nmt-gnmt-6.png)

실제 번역 품질을 측정하기 위하여 BLEU 이외에도 implicit(human) evaluation을 통하여 GNMT의 성능 개선의 정도를 측정하였습니다. 0(Poor)에서 6(Perfect)점 사이로 점수를 매겨 사람의 번역 결과 점수를 Upper Bound로 가정하고 성능의 개선폭을 계산하였습니다. 실제 SMT 방식 대비 엄청난 천지개벽 수준의 성능 개선이 이루어진 것을 알 수 있고, 일부 언어쌍에 대해서는 거의 사람의 수준에 필적하는 성능을 보여주는 것을 알 수 있습니다.

