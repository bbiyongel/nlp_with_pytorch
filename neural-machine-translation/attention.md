# Attention

## ![](/assets/seq2seq_with_attn_architecture.png)

## 소개

### Key-Value function

Attention을 본격 소개하기 전에 먼저 우리가 알고 있는 자료형을 짚고 넘어갈까 합니다. Key-Value 또는 [Python에서 Dictionary](https://wikidocs.net/16)라고 부르는 자료형 입니다.

```
>>> dic = {'dog': 1, 'computer': 2, 'cat': 3}
```

위와 같이 ***Key***와 ***Value***에 해당하는 값들을 넣고 ***Key***를 통해 ***Value*** 값에 접근 할 수 있습니다. 좀 더 바꿔 말하면, ***Query***가 주어졌을 때, ***Key***값에 따라 ***Value***값에 접근 할 수 있습니다. 위의 작업을 함수로 나타낸다면, 아래와 같이 표현할 수 있을겁니다. (물론 실제 Python Dictionary 동작은 매우 다릅니다.)

```
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

코드를 살펴보면, 순차적으로 ***dic*** 내부의 key값들과 ***query*** 값을 비교하여, key가 같을 경우 ***weights***에 ***1.0***을 추가하고, 다를 경우에는 ***0.0***을 추가합니다. 그리고 다시 ***dic*** 내부의 value값들과 weights의 값을 inner product (스칼라곱, dot product) 합니다. 즉, $$ weight = 1.0 $$ 인 경우에만 value 값을 ***answer***에 더합니다.

### Differentiable Key-Value func

좀 더 발전시켜서, 만약 ***is_same*** 함수 대신에 다른 함수를 써 보면 어떻게 될까요? ***how_similar***라는 key와 query 사이의 유사도를 리턴 해 주는 가상의 함수가 있다고 가정해 봅시다. (가정하는 김에 좀 더 가정해서 cosine similarity라고 가정해 봅시다.)

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

무슨 의미인지는 모르겠지만 ***3.2***라는 값이 나왔습니다. 

### Differentiable Key-Value vector function

- 만약, ***dic***의 ***value***에는 100차원의 voctor로 들어있었다면 어떻게 될까요? 
- 거기에, ***query***와 ***key***값 모두 vector라면 어떻게 될까요? 즉, Word Embedding Vector라면?
- 그리고, ***dic***의 ***key***값과 ***value***값이 서로 같다면 어떻게 될까요?

그럼 다시 가상의 함수를 만들어보겠습니다. ***word2vec***이라는 함수는 단어를 입력으로 받아서 그 단어에 해당하는 미리 정해진 word embedding vector를 리턴 해 준다고 가정하겠습니다. 그럼 좀 전의 ***how_similar*** 함수는 두 vector 간의 dot product 값을 반환 할 겁니다.

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

이번에 key_value_func는 그럼 그 값을 받아서 weights에 저장 한 후, 모든 weights의 값이 채워지면 softmax를 취할 겁니다.

```
>>> word2vec('dog')
[0.1, 0.3, -0.7, 0.0, ...
>>> word2vec('cat')
[0.15, 0.2, -0.3, 0.8, ...
>>> dic = {word2vec('dog'): word2vec('dog'), word2vec('computer'): word2vec('computer'), word2vec('cat'): word2vec('cat')}
>>>
>>> query = 'puppy'
>>> answer = key_value_func(word2vec(query))
```

자, 그럼 이제 answer의 값에는 어떤 vector 값이 들어 있을 겁니다. 그 vector는 ***puppy*** vector와 ***dog***, ***computer***, ***cat*** vector들의 유사도에 따라서 값이 정해졌을겁니다.

### Linear Transform


## 설명

## 코드
