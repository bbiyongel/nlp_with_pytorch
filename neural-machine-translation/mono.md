# Monolingual Corpora를 활용하여 성능 쥐어짜기

번역 시스템을 훈련하기 위해서는 다량의 Parallel Corpus(병렬 말뭉치)가 필요합니다. 필자의 경험상 대략 300만 문장쌍이 있으면 완벽하지는 않지만 나름 쓸만한 번역기가 나오기 시작합니다. 하지만 인터넷에는 정말 수치로 정의하기도 힘들 정도의 monolingual corpus가 널려 있는데 반해서, 이러한 parallel corpus을 대량으로 얻는 것은 굉장히 어려운 일입니다. 더군다나, 그 양이 monolingual corpus에 비해서 적기 때문에, 언어 정보를 정확하게 담고 있다고 보기도 어렵고, 모든 정보를 담고 있다고 할 수도 없습니다. 이러한 놓치는 정보를 줄이고, 훈련데이터를 값싸게 구하여 성능을 올리고자 하는 것이 이번 섹션에서 다룰 내용 입니다.

## 1. Language Model Ensemble

![https://arxiv.org/pdf/1503.03535.pdf](/assets/nmt_with_lm_ensemble.png)
[[Gulcehre at el.2015]](https://arxiv.org/pdf/1503.03535.pdf)

![https://arxiv.org/pdf/1503.03535.pdf](/assets/nmt_with_lm_ensemble_evaluation.png)
[[Gulcehre at el.2015]](https://arxiv.org/pdf/1503.03535.pdf)

## 2. Empty source-side translation

## 3. Back translation

![https://arxiv.org/pdf/1511.06709.pdf](/assets/nmt_back_translation.png)
[[Sennrich at el.2015]](https://arxiv.org/pdf/1511.06709.pdf)

## 4. Copied translation

이 방식은 [\[Currey et al.2017\] Copied Monolingual Data Improves Low-Resource Neural Machine
Translation](https://kheafield.com/papers/edinburgh/copy_paper.pdf) 에서 제안 되었습니다.

![https://arxiv.org/pdf/1708.00726.pdf](/assets/nmt_copied_translation.png)
[[Sennrich at el.2017]](https://arxiv.org/pdf/1708.00726.pdf)
