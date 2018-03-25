# Attention

## 1. Motivation

한문장으로 Attention을 정의하면 ***Query와 비슷한 값을 가진 Key를 찾아서 그 Value를 얻는 과정*** 입니다. 기존의 Key-Value 방식과 비교하며 attention에 대해서 설명 하겠습니다.

### a. Key-Value function

Attention을 본격 소개하기 전에 먼저 우리가 알고 있는 자료형을 짚고 넘어갈까 합니다. Key-Value 또는 [Python에서 Dictionary](https://wikidocs.net/16)라고 부르는 자료형 입니다.

```python
dic = {'dog': 1, 'computer': 2, 'cat': 3}
```

위와 같이 _**Key**_와 _**Value**_에 해당하는 값들을 넣고 _**Key**_를 통해 _**Value**_ 값에 접근 할 수 있습니다. 좀 더 바꿔 말하면, _**Query**_가 주어졌을 때, _**Key**_값에 따라 _**Value**_값에 접근 할 수 있습니다. 위의 작업을 함수로 나타낸다면, 아래와 같이 표현할 수 있을겁니다. \(물론 실제 Python Dictionary 동작은 매우 다릅니다.\)

```python
def key_value_func(query):
    weights = []

    for key in dic.keys():
        weights += [is_same(key, query)]

    answer = 0

    for weight, value in zip(weights, dic.values()):
        answer += weight * value

    return answer

def is_same(key, query):
    if key == query:
        return 1.
    else:
        return .0
```

코드를 살펴보면, 순차적으로 _**dic**_ 내부의 key값들과 _**query**_ 값을 비교하여, key가 같을 경우 _**weights**_에 _**1.0**_을 추가하고, 다를 경우에는 _**0.0**_을 추가합니다. 그리고 다시 _**dic**_ 내부의 value값들과 weights의 값을 inner product \(스칼라곱, dot product\) 합니다. 즉, $$ weight = 1.0 $$ 인 경우에만 value 값을 _**answer**_에 더합니다.

### b. Differentiable Key-Value function

좀 더 발전시켜서, 만약 _**is\_same**_ 함수 대신에 다른 함수를 써 보면 어떻게 될까요? _**how\_similar**_라는 key와 query 사이의 유사도를 리턴 해 주는 가상의 함수가 있다고 가정해 봅시다. \(가정하는 김에 좀 더 가정해서 cosine similarity라고 가정해 봅시다.\)

```
>>> query = 'puppy'
>>> how_similar('dog', query)
0.9
>>> how_similar('cat', query)
0.7
>>> how_similar('computer', query)
0.1
```

그리고 해당 함수에 **puppy**라는 단어를 테스트 해 보았더니 위와 같은 값들을 리턴해 주었다고 해 보겠습니다. 그럼 아래와 같이 실행 될 겁니다.

```
>>> query = 'puppy'
>>> key_value_func(query)
3.2 # = 0.9 * 1 + 0.1 * 2 + 0.7 * 3
```

무슨 의미인지는 모르겠지만 _**3.2**_라는 값이 나왔습니다. _**is\_same**_ 함수를 쓸 때에는 두 값이 같은지 if문을 통해 검사하고 값을 할당했기 때문에, 미분을 할 수 없었습니다. 하지만, 이제 우리는 key\_value\_func을 미분 할 수 있습니다.

### c. Differentiable Key-Value Vector function

* 만약, _**dic**_의 _**value**_에는 100차원의 vector로 들어있었다면 어떻게 될까요? 
* 거기에, _**query**_와 _**key**_값 모두 vector라면 어떻게 될까요? 즉, Word Embedding Vector라면?
* 그리고, _**dic**_의 _**key**_값과 _**value**_값이 서로 같다면 어떻게 될까요?

그럼 다시 가상의 함수를 만들어보겠습니다. _**word2vec**_이라는 함수는 단어를 입력으로 받아서 그 단어에 해당하는 미리 정해진 word embedding vector를 리턴 해 준다고 가정하겠습니다. 그럼 좀 전의 _**how\_similar**_ 함수는 두 vector 간의 dot product 값을 반환 할 겁니다.

```
def key_value_func(query):
    weights = []

    for key in dic.keys():
        weights += [how_similar(key, query)]    # dot product 값을 채워 넣는다.

    weights = softmax(weights)    # 모든 weight들을 구한 후에 softmax를 계산한다.
    answer = 0

    for weight, value in zip(weights, dic.values()):
        answer += weight * value

    return answer
```

이번에 key\_value\_func는 그럼 그 값을 받아서 weights에 저장 한 후, 모든 weights의 값이 채워지면 softmax를 취할 겁니다. 여기서 softmax는 weights의 합의 크기를 1로 고정시키는 normalization의 역할을 합니다. 따라서 similarity의 총 합에서 차지하는 비율 만큼 weight의 값이 채워질 겁니다.

```
>>> word2vec('dog')
[0.1, 0.3, -0.7, 0.0, ...
>>> word2vec('cat')
[0.15, 0.2, -0.3, 0.8, ...
>>> len(word2vec('computer'))
100
>>> dic = {word2vec('dog'): word2vec('dog'), word2vec('computer'): word2vec('computer'), word2vec('cat'): word2vec('cat')}
>>>
>>> query = 'puppy'
>>> answer = key_value_func(word2vec(query))
```

자, 그럼 이제 answer의 값에는 어떤 vector 값이 들어 있을 겁니다. 그 vector는 _**puppy**_ vector와 _**dog**_, _**computer**_, _**cat**_ vector들의 유사도에 따라서 값이 정해졌을겁니다.

즉, 다시 말해서, 이 함수는 query와 비슷한 key 값을 찾아서 비슷한 정도에 따라서 weight를 나누고, 각 key의 value값을 weight 값 만큼 가져와서 모두 더하는 것 입니다. 이것이 Attention이 하는 역할 입니다.

## 2. Attention for Machine Translation task

### a. Overview

그럼 번역과 같은 task에서 attention은 어떻게 작용할까요? 번역 과정에서는 encoder의 각 time-step 별 output을 Key와 Value로 삼고, 현재 time-step의 decoder output을 Query로 삼아 attention을 취합니다.

* Query: 현재 time-step의 decoder output
* Keys: 각 time-step 별 encoder output
* Values: 각 time-step 별 encoder output

```
>>> context_vector = attention(query = decoder_output, keys = encoder_outputs, values = encoder_outputs)
```

Attention을 추가한 seq2seq의 수식은 아래와 같은부분이 추가/수정 됩니다.

$$
w = softmax({h_{t}^{tgt}}^T W \cdot H^{src})
$$
$$
c = H^{src} \cdot w~~~~~and~c~is~a~context~vector
$$
$$
\tilde{h}_{t}^{tgt}=\tanh(linear_{2hs \rightarrow hs}([h_{t}^{tgt}; c]))
$$
$$
\hat{y}_{t}=softmax(linear_{hs \rightarrow |V_{tgt}|}(\tilde{h}_{t}^{tgt}))
$$
$$
where~hs~is~hidden~size~of~RNN,~and~|V_{tgt}|~is~size~of~output~vocabulary.
$$

원하는 정보를 attention을 통해 encoder에서 획득한 후, 해당 정보를 decoder output과 concatenate하여 $$ tanh $$를 취한 후, softmax 계산을 통해 다음 time-step의 입력이 되는 $$ \hat{y}_{t} $$을 구합니다.

![](/assets/seq2seq_with_attention_architecture.png)

### b. Linear Transformation

이때, 각 input parameter들은 다음을 의미한다고 볼 수 있습니다.

1. decoder\_output: 현재 time-step 까지 번역 된 target language 단어들 또는 문장, 의미
2. encoder\_outputs: 각 time-step 에서의 source language 단어 또는 문장, 의미

사실 추상적이므로 딱 잘라 정의할 수 없습니다. 하지만 분명한건, source language와 target language가 다르다는 것 입니다. 따라서 단순히 dot product를 해 주기보단 source language와 target language 간에 bridge를 하나 놓아주어야 합니다. 그래서 우리는 두 언어의 embedding hyper plane이 선형 관계에 있다고 가정하고, dot product 하기 전에 _**linear transformation**_을 해 줍니다.

![](/assets/attention_linear_transform.png)

위와 같이 꼭 번역 task가 아니더라도, 두 다른 domain 사이의 비교를 위해서 사용합니다.

### c. Why

![](/assets/attention_working_example.png)

왜 Attention이 필요한 것일까요? 기존의 seq2seq는 두 개의 RNN\(encoder와 decoder\)로 이루어져 있습니다. 여기서 압축된 문장의 의미에 해당하는 encoder의 정보를 hidden state \(LSTM의 경우에는 + cell state\)의 vector로 전달해야 합니다. 그리고 decoder는 그 정보를 이용해 다시 새로운 문장을 만들어냅니다. 이 때, hidden state만으로는 문장의 정보를 완벽하게 전달하기 힘들기 때문입니다. 따라서 decoder의 각 time-step 마다, hidden state의 정보에 추가하여 hidden state의 정보에 따라 필요한 encoder의 정보에 access하여 끌어다 쓰겠다는 것 입니다.

### d. Evaluation

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/attention_better_translation_of_long_sentence.png)  
[Image from CS224n](http://web.stanford.edu/class/cs224n/syllabus.html)

기존 vanila seq2seq는 전반적으로 성능이 떨어짐을 알수 있을 뿐만 아니라, 특히 문장이 길어질 수록 성능이 더욱 하락함을 알 수 있습니다. 하지만 이에 비해서 attention을 사용하면 문장이 길어지더라도 성능이 크게 하락하지 않음을 알 수 있습니다.

## 3. Code



