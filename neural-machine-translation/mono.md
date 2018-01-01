# Monolingual Corpora를 활용하여 성능 쥐어짜기

번역 시스템을 훈련하기 위해서는 다량의 Parallel Corpus(병렬 말뭉치)가 필요합니다. 필자의 경험상 대략 300만 문장쌍이 있으면 완벽하지는 않지만 나름 쓸만한 번역기가 나오기 시작합니다. 하지만 인터넷에는 정말 수치로 정의하기도 힘들 정도의 monolingual corpus가 널려 있는데 반해서, 이러한 parallel corpus을 대량으로 얻는 것은 굉장히 어려운 일입니다. 또한, monolingual corpus가 그 양이 많기 때문에 실제 우리가 사용하는 언어의 확률분포에 좀 더 가까울 수 있고, 따라서 언어모델을 구성함에 있어서 훨씬 유리합니다. 이번 섹션은 이러한 값싼 monolingual corpus를 활용하여 Neural Machine Translation system의 성능을 쥐어짜는 방법들에 대해서 다룹니다.

## 1. Language Model Ensemble

![https://arxiv.org/pdf/1503.03535.pdf](/assets/nmt_with_lm_ensemble.png)
[[Gulcehre at el.2015]](https://arxiv.org/pdf/1503.03535.pdf)

이 방법은 Bengio 교수의 연구실에서 쓴 paper인 [[Gulcehre at el.2015]](https://arxiv.org/pdf/1503.03535.pdf)에서 제안 된 방법입니다. Language Model을 explicit하게 ensemble하여 디코더의 성능을 올리고자 시도하였습니다. 두개의 다른 모델을 쓴 ***shallow fusion*** 방법 보다, LM을 Seq2seq에 포함시켜 end2end training을 하여 한개의 모델로 만든 ***deep fusion*** 방법이 좀 더 나은 성능을 나타냈습니다.

![https://arxiv.org/pdf/1503.03535.pdf](/assets/nmt_with_lm_ensemble_evaluation.png)
[[Gulcehre at el.2015]](https://arxiv.org/pdf/1503.03535.pdf)

성능상으로는 뒤에 다룰 내용들보다 성능 상의 gain이 적지만, 그 내용이 방법의 장점은 Monolingual corpus를 전부 활용 할 수 있다는 것입니다.

## 2. Empty source-side translation

아래의 내용들은 전부 [Edinburgh 대학의 Nematus 번역시스템](https://arxiv.org/pdf/1708.00726.pdf)
에서 제안되고 사용된 내용들입니다. 저자인 Rico Sennrich는 explicit하게 LM을 ensemble 하는 대신, decoder로 하여금 monolingual corpus를 학습할 수 있게 하는 방법을 제안하였습니다. 예전 챕터에서 다루었듯이, 디코더는 ***Conditional Neural Network Language Model***이라고 할 수 있는데, source sentence인 $$ X $$를 빈 입력을 넣어줌으로써, (그리고 Attention등을 모두 dropout 시켜 끊어줌으로써) condition을 없애는 것이 이 방법의 핵심입니다. 저자는 이 방법을 사용하면 decoder가 monolingual corpus의 language model을 학습하는 것과 같다고 하였습니다.

## 3. Back translation

그리고 같은 [논문](https://arxiv.org/pdf/1708.00726.pdf)에서 또 다른 방법을 제시하였습니다. 이 방법은 기존의 훈련된 ***반대 방향*** 번역기를 사용하여 monolingual corpus를 기계번역하여 synthetic parallel corpus를 만들어 이것을 훈련에 사용하는 방식 입니다. 중요한 점은 기계번역에 의해 만들어진 synthetic parallel corpus를 사용할 때, 반대방향의 번역기의 훈련에 사용한다는 것 입니다.

예를 들어, ***한국어*** monolingual corpus가 있을 때, 이것을 기존에 훈련된 ***한***$$ \rightarrow $$***영***번역기에 기계번역시켜 한-영 synthetic parallel corpus를 만들고, 이것을 ***영***$$ \rightarrow $$***한***번역기를 훈련시키는데 사용하는 것 입니다. 이러한 방법의 특성 때문에 ***back translation*** 이라고 명명되었습니다.

![https://arxiv.org/pdf/1511.06709.pdf](/assets/nmt_back_translation.png)
[[Sennrich at el.2015]](https://arxiv.org/pdf/1511.06709.pdf)

위의 Table은 Empty source-side translation(==monolingual)과 back translation(==synthetic) 방식에 대해서 성능을 실험한 결과 입니다.

## 4. Copied translation

이 방식은 [\[Currey et al.2017\] Copied Monolingual Data Improves Low-Resource Neural Machine
Translation](https://kheafield.com/papers/edinburgh/copy_paper.pdf) 에서 제안 되었습니다.

![https://arxiv.org/pdf/1708.00726.pdf](/assets/nmt_copied_translation.png)
[[Sennrich at el.2017]](https://arxiv.org/pdf/1708.00726.pdf)
