# 피쳐 추출하기: TF-IDF

본격적으로 피쳐 벡터를 만들기에 앞서, 이번 섹션에서는 텍스트 마이닝(Text Mining)에서 중요하게 사용되는 TF-IDF를 사용하여 피쳐를 추출해 보도록 하겠습니다.

$$\text{TF-IDF}(w,d)=\frac{\text{TF}(w,d)}{\text{DF}(w)}$$

TF-IDF는 출현빈도를 사용하여 어떤 단어 $w$ 가 문서 $d$ 내에서 얼마나 중요한지 나타내는 수치 입니다. 이 수치가 높을 수록 $w$ 는 $d$ 를 대표하는 성격을 띄게 된다고 볼 수도 있습니다. TF-IDF의 TF는 Term Frequency로 단어의 문서 내에 출현한 횟수를 의미 합니다. 그리고 IDF는 Inverse Document Frequency로 그 단어가 출현한 문서의 숫자의 역수(inverse)를 의미 합니다. 다시 설명하면, Document Frequency (DF)는 해당 단어가 출현한 문서의 수가 되고, inverse가 붙어 역수를 취한 것 입니다.

TF는 단어가 문서 내에서 출현한 횟수입니다. 따라서 그 숫자가 클수록 문서 내에서 중요한 단어일 확률이 높습니다. 하지만, 'the'와 같은 단어도 TF값이 매우 클 것입니다. 하지만 'the'가 중요한 경우는 거의 없을 것 입니다. 따라서 이때 IDF가 필요 합니다. DF는 그 단어가 출현한 문서의 숫자를 의미 하기 때문에, 그 값이 클수록 'the'와 같이 일반적으로 많이 쓰이는 단어일 가능성이 높습니다. 따라서 IDF를 구해 TF에 곱해줌으로써, 'the'와 같은 단어들에 대한 패널티를 줍니다. 최종적으로 우리가 얻는 숫자는, 다른 문서들에서는 잘 나타나지 않지만 특정 문서에서만 잘 나타난 경우에 횟수가 높아지기 때문에, 특정 문서에서 얼마나 중요한 역할을 차지하는지 나타내는 수치가 될 수 있습니다.

## TF-IDF 예제

우리는 아래와 같이 간단하게 분절(tokenize)된 여러 문서가 주어졌을 때, 단어들의 TF-IDF를 구하는 방법을 살펴보겠습니다.

먼저 문서가 주어졌을 때, 문서 내의 단어들의 출현 빈도를 세는 함수는 아래와 같이 구현 할 수 있습니다.

```python
import pandas as pd

def get_term_frequency(document, word_dict=None):
    if word_dict is None:
        word_dict = {}
    words = document.split()

    for w in words:
        word_dict[w] = 1 + (0 if word_dict.get(w) is None else word_dict[w])

    return pd.Series(word_dict).sort_values(ascending=False)
```

그리고 문서들이 주어졌을 때, 각 단어들이 몇개의 문서에서 나타났는지 세는 함수는 아래와 같이 구현 할 수 있습니다.

```python
def get_document_frequency(documents):
    dicts = []
    vocab = set([])
    df = {}

    for d in documents:
        tf = get_term_frequency(d)
        dicts += [tf]
        vocab = vocab | set(tf.keys())

    for v in list(vocab):
        df[v] = 0
        for dict_d in dicts:
            if dict_d.get(v) is not None:
                df[v] += 1

    return pd.Series(df).sort_values(ascending=False)
```

그럼 아래와 같은 이름으로 위의 문서들이 각 변수에 들어있다고 가정해 보도록 하겠습니다.

```python
doc1, doc2, doc3
```

그럼 TF-IDF를 계산하는 최종 함수는 아래와 같이 구현할 수 있을 것 입니다.

```python
def get_tfidf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frequency(d, vocab)
        tfs += [get_term_frequency(d)]
    df = get_document_frequency(docs)

    from operator import itemgetter
    import numpy as np

    stats = []
    for word, freq in vocab.items():
        tfidfs = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tfidfs += [tfs[idx][word] * np.log(len(docs) / df[word])]
            else:
                tfidfs += [0]

        stats.append((word, freq, *tfidfs, max(tfidfs)))

    return pd.DataFrame(stats, columns=('word',
                                        'frequency',
                                        'doc1',
                                        'doc2',
                                        'doc3',
                                        'max')).sort_values('max', ascending=False)
```

```python
>>> get_tfidf([doc1, doc2, doc3])
```

위의 코드를 차례대로 실행한 결과는 아래와 같습니다. 첫번째 문서에서 가장 중요한 단어는 '남자'임을 알 수 있고, 마찬가지로 두번째 문서는 '요인'인 것을 알 수 있습니다.

|단어( $w$ )|총 출현 횟수|TF-IDF( $d_1$ )|TF-IDF( $d_2$ )|TF-IDF( $d_3$ )|
|-|-|-|-|-|
|남자|9|9.8875|0.0000|0.0000|
|요인|6|0.0000|6.5917|0.0000|
|심리학|5|5.4931|0.0000|0.0000|
|멀리|4|4.3944|0.0000|0.0000|
|시험|4|0.0000|4.3944|0.0000|
|환경|4|0.0000|4.3944|0.0000|
|성|4|0.0000|4.3944|0.0000|
|었|4|0.0000|0.0000|4.3944|
|제|4|0.0000|0.0000|4.3944|
|대해|3|3.2958|0.0000|0.0000|
|나|3|3.2958|0.0000|0.0000|
|간|3|3.2958|0.0000|0.0000|
|유형|3|0.0000|3.2958|0.0000|
|됩니다|3|0.0000|3.2958|0.0000|
|을까요|3|0.0000|3.2958|0.0000|
|인증|3|0.0000|3.2958|0.0000|
|탓|3|0.0000|3.2958|0.0000|
|유전자|3|0.0000|3.2958|0.0000|
|유전|3|0.0000|3.2958|0.0000|
|쌍둥이|3|0.0000|3.2958|0.0000|
|경우|3|0.0000|3.2958|0.0000|
