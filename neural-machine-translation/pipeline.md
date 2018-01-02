# Pipeline for Machine Translation

이번 섹션에서는 실제 Machine Translation 시스템을 구축하기 위한 절차와 시스템 Pipeline이 어떻게 구성되는지 살펴보도록 하겠습니다. 아울러 그 다음에는 실제 Google 등에서 발표한 논문을 통해서 그들의 상용 번역 시스템의 실제 구성에 대해서 살펴보도록 하겠습니다.

통상적으로 번역시스템을 구축하면 아래와 같은 흐름을 가지게 됩니다.

1. Corpus 수집, 정제
1. Tokenization
1. Batchfy
1. Training
1. Inference
1. Tokenization 복원