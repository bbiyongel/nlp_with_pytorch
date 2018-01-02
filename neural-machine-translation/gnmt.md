# Google Neural Machine Translation \(GNMT\)

Google은 2016년 논문([\[Wo at el.2016\]](https://arxiv.org/pdf/1609.08144.pdf)
)을 발표하여 그들의 번역시스템에 대해서 상세히 소개하였습니다. 실제 시스템에 적용된 모델 architecture부터 훈련 algorithm 까지 상세히 기술하였기 때문에, 실제 번역 시스템을 구성하고자 할 때에 좋은 reference가 될 수 있습니다. 또한 다른 논문들에서 실험 결과에 대해 설명할 때, GNMT를 upper boundary baseline으로 참조하기도 합니다.

## 1. Model Architecture

### a. Residual Connection

![](/assets/nmt-gnmt-1.png)

### b. Bi-directional Encoder for First Layer

![](/assets/nmt-gnmt-2.png)

## 2. Segmentation Approachs

### a. Wordpiece Model

### b. Mixed Word/Character Model

## 3. Training Criteria

![](/assets/nmt-gnmt-5.png)

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



