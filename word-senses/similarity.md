# Using Other Features for Similarity

이번엔 다른 방식으로 단어의 유사도를 구하는 방법에 접근 해보겠습니다. 자체적으로 단어에 대한 특성(feature)들을 모아 feature vector로 만들거나 유사도(similarity)를 계산하는 연산을 통해 단어간의 유사도를 구하는 방법입니다. 지금이야 어렵지않게 단어를 vector 형태로 embedding 할 수 있지만, 딥러닝 이전의 시대에는 쉽지 않은 일이었습니다. 이번 섹션을 통해서 딥러닝 이전의 전통적인 방식의 단어간의 유사도를 구하는 방법에 대해 알아보고, 이 방법의 단점과 한계에 대해서 살펴보겠습니다.

## Based on Co-Occurrence

가장 쉽게 먼저 생각할 수 있는 방식은 함께 나타나는 단어들을 활용한 방법 입니다. 의미가 비슷한 단어라면 쓰임새가 비슷할 것 입니다. 또한, 쓰임새가 비슷하기 때문에, 비슷한 문장 안에서 비슷한 역할로 사용될 것이고, 따라서 함께 나타나는 단어들이 유사할 것 입니다. 이러한 관점에서 우리는 함께 나타나는 단어들이 유사한 단어들의 유사도를 높게 주도록 만들어 줄 것 입니다.

https://web.stanford.edu/class/cs124/lec/sem

https://www.cs.princeton.edu/courses/archive/fall16/cos402/lectures/402-lec10.pdf

### Term-Frequency Matrix

앞서 우리는 TF-IDF에 대해서 살펴 보았습니다. TF-IDF에서 사용되었던 TF (term frequency)는 훌륭한 피쳐(feature)가 될 수 있습니다. 예를 들어 어떤 단어가 각 문서별로 출현한 횟수가 차원별로 구성되면, 하나의 feature vector를 이룰 수 있습니다. 물론 각 문서별 TF-IDF 자체를 사용하는 것도 가능합니다.

```python
vocab = {}
tfs = []
for d in docs:
    vocab = get_term_frequency(d, vocab)
    tfs += [get_term_frequency(d)]

from operator import itemgetter
import numpy as np
sorted_vocab = sorted(vocab.items(), key=itemgetter(1), reverse=True)

stats = []
for v, freq in sorted_vocab:
    tf_v = []
    for idx in range(len(docs)):
        if tfs[idx].get(v) is not None:
            tf_v += [tfs[idx][v]]
        else:
            tf_v += [0]

    print('%s\t%d\t%s' % (v, freq, '\t'.join(['%d' % tf for tf in tf_v])))
```

위의 코드를 사용하여 단어들의 각 문서별 출현횟수를 나타내면 아래와 같습니다.

|단어($w$)|TF($w$)|TF($w,d_1$)|TF($w,d_2$)|TF($w,d_3$)|
|-|-|-|-|-|
|는|47|15|14|18|
|을|39|8|10|21|
|.|36|16|10|10|
|하|33|10|9|14|
|이|32|8|8|16|
|들|31|14|7|10|
|의|27|9|15|3|
|,|26|10|5|11|
|를|20|8|6|6|
|에|19|6|6|7|
|것|17|6|4|7|
|적|17|2|9|6|
|은|15|6|2|7|
|있|14|4|7|3|
|가|14|1|7|6|
|여러분|12|5|6|1|
|말|11|5|1|5|
|한|11|5|3|3|
|수|11|3|7|1|
|고|10|4|4|2|
|게|10|2|7|1|

여기서 마지막 3개 column이 각 단어별 문서에 대한 출현횟수를 활용한 feature vector가 될 것 입니다.

## Pointwise Mutual Information

### PMI between two words

### Positive PMI (PPMI)

## Cosine Similarity

## Jaccard Similarity

## Appendix: Similarity between Documents