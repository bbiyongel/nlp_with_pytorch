# The University of Edinburgh’s Neural MT Systems for WMT17

사실 Google의 논문은 훌륭하지만 매우 스케일이 매우 큽니다. 저는 그래서 작은 스케일의 기계번역 시스템에 관한 논문은 이 논문[\[Sennrich at el.2017\]](https://arxiv.org/pdf/1708.00726.pdf)을 높게 평가합니다. 이 논문도 기계번역 시스템을 구성할 때에 훌륭한 baseline이 될 수 있습니다.
Edinburgh 대학의 Sennrich교수는 매년 열리는 WMT 대회에 참가하고 있고, 해당 대회에 참가하는 기계번역 시스템들은 이처럼 매년 자신들의 기술에 대한 논문을 제출합니다. 좋은 참고자료로 삼을 수 있습니다.

## Subword Segmentation

![](/assets/nmt-edinburgh-1.png)
[[Sennrich at el.2016]](http://www.aclweb.org/anthology/P16-1162)

이 논문 또한 (그들이 처음으로 제안한 방식이기에) BPE 방식을 사용하여 tokenization을 수행하였습니다. 이제 우리는 subword 기반의 tokenization 방식이 하나의 정석이 되었음을 알 수 있습니다. 위의 code는 BPE algorithm에 대해서 간략하게 소개한 code 입니다. 전처리 챕터에서 소개했지만, subword 방식은 위와 같이 가장 많이 등장한 문자열(character sequence)에 대해서 합쳐주며 iteration을 반복하고, 원하는 어휘(vocabulary) 숫자가 채워질때가지 해당 iteration을 반복합니다.

## Architecture

이 논문에서는 seq2seq를 기반으로 모델 구조(architecture)를 만들었는데, 다만 LSTM이 아닌 GRU를 사용하여 RNN stack을 구성하였습니다. Google과 마찬가지로 residual connection을 사용하여 stack을 구성하였고, encoder의 경우에는 4개층, decoder의 경우에는 8개 층을 쌓아 모델을 구성하였습니다. 실험 시에는 $$ hidden~size = 1024,~word~vector~dimension = 512 $$를 사용하였습니다. 또한, Google과는 다르게 순수하게 Adam만을 optimizer로 사용하여 훈련을 하였습니다.

## Synthetic Data using Monolingual Data

이전 섹션에서 소개한 그들이 제안한 논문[[Sennrich at el.2015]](https://arxiv.org/pdf/1511.06709.pdf)의 방식대로 back translation과 copied translation 방식을 사용하여 합성 병렬(pseudo parallel) corpus를 구성하여 훈련 데이터셋에 추가하였습니다. 이때에 비율은 실험결과에 따라서 $$ parallel : copied : back = 1 : 1 \sim 2 : 1\sim 2 $$로 조절하여 사용하였습니다.

## Ensemble

이 논문에서 그들은 2가지 앙상블(ensemble) 기법을 모두 사용하였습니다.

- checkpoint ensemble
    + 특정 epoch에서부터 다른 모델로 다시 훈련하여 ensemble을 구성합니다. 훈련 중간부터 다시 훈련하기 때문에 시간적으로 굉장히 효율적입니다.
- independent ensemble
    + 처음부터 다른 모델로 훈련하여 ensemble로 구성합니다. 처음부터 다시 훈련하므로 checkpoint 방식에 비해서 비효율적이지만, diversity 관점에서 낫습니다.