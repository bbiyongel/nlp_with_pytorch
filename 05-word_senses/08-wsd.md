# 단어 중의성 해소 (Word Sense Disambiguation, WSD)

이번 섹션에서는 단어의 유사도가 아닌, 모호성에 대해 다루어 보겠습니다. 하나의 단어는 여러가지 의미를 지닐 수 있습니다. 비슷한 의미의 동형이의어(homonym)인 경우에는 비교적 그 문제가 크지 않을 수 있습니다. 조금은 어색한 문장 표현이나 이해가 될 수 있겟지만, 큰 틀에서는 벗어나지 않기 때문입니다. 하지만 다의어(polysemy)의 경우에는 문제가 커집니다. 앞서 예를 들었던 '차'의 경우에 'tea'의 의미로 해석되느냐, 'car'의 의미로 해석되느냐에 따라 문장의 의미가 매우 달라질것이기 때문입니다. 따라서 이와 같이 단어가 주어졌을 때, 그 의미의 모호성을 없애고 해석하는 것이 매우 중요합니다. 이를 단어 중의성 해소(word sense disambiguation)이라고 합니다.

|원문|차를 마시러 공원에 가던 차 안에서 나는 그녀에게 차였다.|
|-|-|
|G*|I was kicking her in the car that went to the park for tea.|
|M*|I was a car to her, in the car I had a car and went to the park.|
|N*|I got dumped by her on the way to the park for tea.|
|K*|I was in the car going to the park for tea and I was in her car.|
|S*|I got dumped by her in the car that was going to the park for a cup of tea.|

위와 같이 다양한 '차'가 문장에 나타났을 때, 실제 기계번역에 있어서 심각한 성능저하의 원인이 되기도 합니다.

## 시소러스 기반 중의성 해소: Lesk 알고리즘

Lesk 알고리즘은 가장 간단한 사전 기반 중의성 해소 방법입니다. 주어진 문장에서 특정 단어에 대해서 의미를 명확히 하고자 할 때 사용 할 수 있습니다. 이를 위해서 Lesk 알고리즘은 간단한 가정을 하나 만듭니다. 문장 내에 같이 등장하는 단어(context)들은 공통 토픽을 공유한다는 것 입니다.

이 가정을 바탕으로 동작하는 Lesk 알고리즘의 개요는 다음과 같습니다. 먼저, 중의성을 해소하고자 하는 단어에 대해서 사전(주로 워드넷)의 의미별 설명과 주어진 문장 내에 등장한 단어의 사전에서 의미별 설명 사이의 유사도를 구합니다. 유사도를 구하는 방법은 여러가지가 있을테지만, 가장 간단한 방법으로는 겹치는 단어의 갯수를 구하는 것이 될 수 있습니다. 이후, 문장 내 단어들의 의미별 설명과 가장 유사도가 높은 (또는 겹치는 단어가 많은) 의미가 선택 됩니다.

예를 들어 NLTK의 워드넷에 'bass'를 검색해 보면 아래와 같습니다.

```python
>>> from nltk.corpus import wordnet as wn
>>> for ss in wn.synsets('bass'):
...     print(ss, ss.definition())
...
Synset('bass.n.01') the lowest part of the musical range
Synset('bass.n.02') the lowest part in polyphonic music
Synset('bass.n.03') an adult male singer with the lowest voice
Synset('sea_bass.n.01') the lean flesh of a saltwater fish of the family Serranidae
Synset('freshwater_bass.n.01') any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)
Synset('bass.n.06') the lowest adult male singing voice
Synset('bass.n.07') the member with the lowest range of a family of musical instruments
Synset('bass.n.08') nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes
Synset('bass.s.01') having or denoting a low vocal or instrumental range
```

Lesk 알고리즘 수행을 위하여 간단하게 랩핑(wrapping)하도록 하겠습니다.

```python
def lesk(sentence, word):
    from nltk.wsd import lesk

    best_synset = lesk(sentence.split(), word)
    print(best_synset, best_synset.definition())
```

아래와 같이 주어진 문장에서는 'bass'는 물고기의 의미로 뽑히게 되었습니다.

```python
>>> sentence = 'I went fishing last weekend and I got a bass and cooked it'
>>> word = 'bass'
>>> lesk(sentence, word)

Synset('sea_bass.n.01') the lean flesh of a saltwater fish of the family Serranidae
```

또한, 아래의 문장애서는 음악에서의 역할을 의미하는 것으로 추정되었습니다.

```python
>>> sentence = 'I love the music from the speaker which has strong beat and bass'
>>> word = 'bass'
>>> lesk(sentence, word)

Synset('bass.n.02') the lowest part in polyphonic music
```

여기까지 보면 Lesk 알고리즘은 비교적 잘 동작하는 것으로 보입니다. 하지만, 비교적 정확하게 예측해낸 위의 두 사례와 달리, 아래의 문장에서는 전혀 다른 의미로 예측하는 것을 볼 수 있습니다.

```python
>>> sentence = 'I think the bass is more important than guitar'
>>> word = 'bass'
>>> lesk(sentence, word)

Synset('sea_bass.n.01') the lean flesh of a saltwater fish of the family Serranidae
```

이처럼 Lesk 알고리즘은 명확한 장단점을 지니고 있습니다. 워드넷과 같이 잘 분류된 사전이 있다면, 쉽고 빠르게 중의성해소를 해결할 수 있을 것 입니다. 또한 워드넷에서 이미 단어별로 몇개의 의미를 갖고 있는지 잘 정의 해 놓았기 때문에, 크게 고민 할 필요도 없습니다. 하지만 보다시피 사전의 단어 및 의미에 대한 설명에 크게 의존하게 되고, 설명이 부실하거나 주어진 문장에 큰 특징이 없을 경우 단어 중의성해소 능력은 크게 떨어지게 됩니다. 그리고 워드넷이 모든 언어에 대해서 존재하는 것이 아니기 때문에, 사전이 존재하지 않는 언어에 대해서는 Lesk 알고리즘 자체의 수행이 어려울 수도 있습니다.
