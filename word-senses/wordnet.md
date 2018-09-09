# Word Sense from Thesaurus

단어 중의성해소(WSD)에는 여러가지 방법이 있습니다. 첫 번째 다룰 방법은 어휘 분류 사전(thesaurus, 시소러스)을 활용한 방법 입니다. 기존에 구축 된 사전은 해당 단어에 대한 자세한 풀이와 의미, 동의어 등의 자세한 정보를 담고 있기 마련입니다. 이렇게 기존에 구축된 공개되어 있는 사전을 활용하여 중의성을 해소하고자 하는 방법 입니다.

## WordNet

[WordNet](https://wordnet.princeton.edu/)(워드넷)은 심리학 교수인 George Armitage Miller 교수가 지도하에 프린스턴 대학에서 1985년 부터 만들어진 프로그램 입니다. 처음에는 주로 기계번역(Machine Translation)을 돕기 위한 목적으로 만들어졌으며, 따라서 동의어 집합(Synset) 또는 상위어(Hypernym)나 하위어(Hyponym)에 대한 정보가 특히 잘 구축되어 있는 것이 장점 입니다. 단어에 대한 상위어와 하위어 정보를 구축하게 됨으로써, Directed Acyclic Graph(유방향 비순환 그래프)를 이루게 됩니다. (Tree구조가 아닌 이유는 하나의 노드가 여러 상위 노드를 가질 수 있기 때문입니다.)

![각 단어별 top-1 sense의 top-1 hypernym만 선택하여 tree로 나타낸 경우](../assets/wsd-wordnet-hierarchy.png)

WordNet은 프로그램으로 제공되므로 다운로드 받아 설치할 수도 있고, [웹사이트](http://wordnetweb.princeton.edu/perl/webwn)에서 바로 이용 할 수도 있습니다. 또한, NLTK에 랩핑(wrapping)되어 포함되어 있어, import하여 사용 가능합니다. 아래는 WordNet 웹사이트에서 bank를 검색한 결과 입니다.

![[WordNet 웹사이트](http://wordnetweb.princeton.edu/perl/webwn)에서 단어 'bank'를 검색 한 결과](../assets/wsd-wordnet-screenshot.png)

그림에서와 같이 bank라는 단어에 대해서 명사(noun)일때의 의미 10개, 동사(verb)인 경우의 의미 8개를 정의 해 놓았습니다. 명사 bank#2의 경우에는 여러 다른 표현(depository finaancial institution#1, banking concern#1)들도 같이 게시되어 있는데, 이것은 

이처럼 WordNet은 단어 별 여러가지 가능한 의미를 미리 정의 하고, numbering 해 놓았습니다. 또한 각 의미별로 비슷한 의미를 갖는 동의어(Synonym)를 링크 해 놓아, Synset을 제공합니다. 이것은 단어 중의성 해소에 있어서 매우 좋은 labeled data가 될 수 있습니다. 만약 WordNet이 없다면, 각 단어별로 몇 개의 의미가 있는지조차 알 수가 없을 것입니다. 죽, WordNet 덕분에 우리는 이 데이터를 바탕으로 supervised learning(지도 학습)을 통해 단어 중의성 해소 문제를 풀 수 있습니다.

## 한국어 WordNet

다행히 영어 WordNet과 같이 한국어를 위한 WordNet도 존재 합니다. 다만, 아직까지 표준이라고 할만큼 정해진 것은 없고 몇 개의 WordNet이 존재하고 있습니다. 아직까지 지속적으로 발전하고 있는 만큼, 작업에 따라 필요한 한국어 WordNet을 이용하면 좋습니다.

|이름|기관|웹사이트|
|-|-|-|
|KorLex|부산대학교|http://korlex.pusan.ac.kr/|
|Korean WordNet(KWN)|KAIST|http://wordnet.kaist.ac.kr/|

## Word Similarity Based on Path in WordNet

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

위의 코드를 사용하면 WordNet에서 특정 단어의 최상위 부모 노드까지의 경로를 구할 수 있습니다. 아래와 같이 'policeman'은 'firefighter', 'sheriff'와 매우 비슷한 경로를 가짐을 알 수 있습니다. 'student'와도 매우 비슷하지만, 'mailman'과 좀 더 비슷함을 알 수 있습니다.

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

위로 부터 얻어낸 정보들을 취합하여 그래프로 나타내면 아래와 같습니다.

![](../assets/wsd-wordnet-hierarchy.png)

각 leaf 노드들은 query 단어들이 됩니다. 우리는 여기서 각 leaf 노드 간의 최단 거리를 유사도 정보로 활용할 수 있습니다.

$$
\text{similarity}(w, w')=-\log{\text{distance(w, w')}}
$$