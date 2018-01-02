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

### b. Mixed Word/Character Model

## 3. Training Criteria

![](/assets/nmt-gnmt-5.png)

Google은 후에 설명할 Reinforcement Learning 기법을 사용하여 Maximum Likelihood Estimation (MLE)방식의 훈련된 모델에 fine-tuning을 수행하였습니다. 따라서 위의 테이블과 같은 추가적이 성능 개선을 얻어낼 수 있었습니다. 이러한 RL 기법은 다음 챕터에서 소개하도록 하겠습니다.

## 4. Quantization

![](/assets/nmt-gnmt-3.png)

## 5. Optimizer

![](/assets/nmt-gnmt-4.png)

## 6. Decoder

### a. Length Penalty

### b. Coverage Penalty

## 7. Training Procedure

## 8. Evaluation

![](/assets/nmt-gnmt-6.png)



