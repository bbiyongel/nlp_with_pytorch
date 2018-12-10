# 시소러스를 활용한 단어 의미 파악

앞선 섹션에서, 단어는 내부에 의미를 지니고 있고 그 의미는 개념과 같아서 계층적 구조를 지닌다고 하였습니다. 아쉽게도 one-hot 벡터로는 그런 단어 의미의 특성을 잘 반영할 수 없었습니다. 만약 그 계층 구조를 잘 분석하고 분류하여 데이터베이스로 구축한다면, 우리가 자연어처리를 하고자 할 때 매우 큰 도움이 될 것 입니다. 이런 용도로 구축된 데이터베이스를 시소러스(thesaurus, 어휘분류사전)라고 부릅니다. 이번 섹션에서는 시소러스의 대표인 워드넷(WordNet)에 대해 다루어 보겠습니다.

## 워드넷(WordNet)

[워드넷(WordNet)](https://wordnet.princeton.edu/)은 1985년부터 심리학 교수인 George Armitage Miller 교수의 지도하에 프린스턴 대학에서 만든 프로그램 입니다. 처음에는 주로 기계번역(Machine Translation)을 돕기 위한 목적으로 만들어졌으며, 따라서 동의어 집합(Synset) 또는 상위어(Hypernym)나 하위어(Hyponym)에 대한 정보가 특히 잘 구축되어 있는 것이 장점 입니다. 단어에 대한 상위어와 하위어 정보를 구축하게 됨으로써, 유방향 비순환 그래프(Directed Acyclic Graph를 이루게 됩니다. (트리 구조가 아닌 이유는 하나의 노드가 여러 상위 노드를 가질 수 있기 때문입니다.)

![각 단어별 top-1 의미의 top-1 상위어만 선택하여 트리 구조로 나타낸 경우](../assets/wsd-wordnet_hierarchy.png)

워드넷은 프로그램으로 제공되므로 다운로드 받아 설치할 수도 있고, [웹사이트](http://wordnetweb.princeton.edu/perl/webwn)에서 바로 이용 할 수도 있습니다. 또한, NLTK에 랩핑(wrapping)되어 포함되어 있어, import하여 사용 가능합니다. 아래는 워드넷 웹사이트에서 'bank'를 검색한 결과 입니다.

![[워드넷 웹사이트](http://wordnetweb.princeton.edu/perl/webwn)에서 단어 'bank'를 검색 한 결과](../assets/wsd-wordnet_screenshot.png)

그림에서와 같이 'bank'라는 단어에 대해서 명사(noun)일때의 의미 10개, 동사(verb)인 경우의 의미 8개를 정의 해 놓았습니다. 명사 'bank#2'의 경우에는 여러 다른 표현(depository finaancial institution#1, banking concern#1)들도 같이 게시되어 있는데, 이것이 동의어 집합(synset) 입니다.

이처럼 워드넷은 단어 별 여러가지 가능한 의미를 미리 정의 하고, numbering 해 놓았습니다. 또한 각 의미별로 비슷한 의미를 갖는 동의어(Synonym)를 링크 해 놓아, 동의어 집합(Synset)을 제공합니다. 이것은 단어 중의성 해소에 있어서 매우 좋은 레이블 데이터가 될 수 있습니다. 만약 워드넷이 없다면, 각 단어별로 몇 개의 의미가 있는지조차 알 수가 없을 것입니다. 죽, 워드넷 덕분에 우리는 이 데이터를 바탕으로 지도 학습(supervised learning)을 통해 단어 중의성 해소 문제를 풀 수 있습니다.

## 한국어 워드넷

다행히 영어 워드넷과 같이 한국어를 위한 워드넷도 존재 합니다. 다만, 아직까지 표준이라고 할만큼 정해진 것은 없고 몇 개의 워드넷이 존재하고 있습니다. 아직까지 지속적으로 발전하고 있는 만큼, 작업에 따라 필요한 한국어 워드넷을 이용하면 좋습니다.

|이름|기관|웹사이트|
|-|-|-|
|KorLex|부산대학교|http://korlex.pusan.ac.kr/|
|Korean WordNet(KWN)|KAIST|http://wordnet.kaist.ac.kr/|

## 워드넷을 활용한 단어간 유사도 비교

```python
from nltk.corpus import wordnet as wn

def get_hypernyms(synset):
    current_node = synset
    while True:
        print(current_node)
        hypernym = current_node.hypernyms()
        if len(hypernym) == 0:
            break
        current_node = hypernym[0]
```

위의 코드를 사용하면 워드넷에서 특정 단어의 최상위 부모 노드까지의 경로를 구할 수 있습니다. 아래와 같이 'policeman'은 'firefighter', 'sheriff'와 매우 비슷한 경로를 가짐을 알 수 있습니다. 'student'와도 매우 비슷하지만, 'mailman'과 좀 더 비슷함을 알 수 있습니다.

```python
>>> get_hypernyms(wn.synsets('policeman')[0])
Synset('policeman.n.01')
Synset('lawman.n.01')
Synset('defender.n.01')
Synset('preserver.n.03')
Synset('person.n.01')
Synset('causal_agent.n.01')
Synset('physical_entity.n.01')
Synset('entity.n.01')
```

```python
>>> get_hypernyms(wn.synsets('firefighter')[0])
Synset('fireman.n.04')
Synset('defender.n.01')
Synset('preserver.n.03')
Synset('person.n.01')
Synset('causal_agent.n.01')
Synset('physical_entity.n.01')
Synset('entity.n.01')
```

```python
>>> get_hypernyms(wn.synsets('sheriff')[0])
Synset('sheriff.n.01')
Synset('lawman.n.01')
Synset('defender.n.01')
Synset('preserver.n.03')
Synset('person.n.01')
Synset('causal_agent.n.01')
Synset('physical_entity.n.01')
Synset('entity.n.01')
```

```python
>>> get_hypernyms(wn.synsets('mailman')[0])
Synset('mailman.n.01')
Synset('deliveryman.n.01')
Synset('employee.n.01')
Synset('worker.n.01')
Synset('person.n.01')
Synset('causal_agent.n.01')
Synset('physical_entity.n.01')
Synset('entity.n.01')
```

```python
>>> get_hypernyms(wn.synsets('student')[0])
Synset('student.n.01')
Synset('enrollee.n.01')
Synset('person.n.01')
Synset('causal_agent.n.01')
Synset('physical_entity.n.01')
Synset('entity.n.01')
```

위로 부터 얻어낸 정보들을 취합하여 그래프로 나타내면 아래와 같습니다. 그림에서 각 최하단 노드들은 코드에서 쿼리로 주어진 단어들이 됩니다. 

![각 단어들의 쿼리 결과 구조도](../assets/wsd-wordnet_hierarchy.png)

이때 각 노드들간에 거리를 우리는 구할 수 있습니다. 아래의 그림에 따르면 'student'에서 'fireman'으로 가는 최단거리에는 'enrollee', 'person', 'preserver', 'defender' 노드들이 위치하고 있습니다. 따라서 'student'와 'fireman'의 거리는 5임을 알 수 있습니다.

!['student'와 'fireman' 사이에 위치한 노드들(점선)](../assets/wsd-wordnet_distance.png)

이처럼 우리는 각 최하단 노드 간의 최단 거리를 알 수 있고, 이것을 유사도로 치환하여 활용할 수 있습니다. 당연히 거리가 멀수록 단어간의 유사도는 떨어질테니, 아래와 같은 공식을 적용 해 볼 수 있습니다.

$$
\text{similarity}(w, w')=-\log{\text{distance}(w, w')}
$$

위와 같이 시소러스(thesaurus) 기반의 정보를 활용하여 단어간의 유사도를 구할 수 있습니다. 하지만 사전을 구축하는데는 너무 많은 비용과 시간이 소요 됩니다. 또한, 아무 사전이나 되는 것이 아닌, 상위어(hypernym)와 하위어(hyponym)가 잘 반영되어 있어야 할 것 입니다. 이처럼 사전에 기반한 유사도를 구하는 방식은 비교적 정확한 값을 구할 수 있으나, 그 한계가 뚜렷합니다.