# Attention

## ![](/assets/seq2seq_with_attn_architecture.png)

## 소개

### Key-Value function

Attention을 본격 소개하기 전에 먼저 우리가 알고 있는 자료형을 짚고 넘어갈까 합니다. Key-Value 또는 [Python에서 Dictionary](https://wikidocs.net/16)라고 부르는 자료형 입니다.

```
dic = {'dog': 1, 'computer': 2, 'cat': 3}
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

좀 더 발전시켜서, 만약 ***is_same*** 함수 대신에 다른 함수를 써 보면 어떻게 될까요? ***how_similar***라는 key와 query 사이의 유사도를 리턴 해 주는 가상의 함수가 있다고 가정해 봅시다. (가정하는 김에 좀 더 가정해서 cosine similarity라고 가정해 봅시다.)

```
>>> query = 'puppy'
>>> is_similar('dog', query)
0.9
>>> is_similar('cat', query)
0.7
>>> is_similar('computer', query)
0.1
```

그리고 해당 함수에 **puppy**라는 단어를 테스트 해 보았더니 위와 같은 값들을 리턴해 주었다고 해 보겠습니다. 그럼 아래와 같이 실행 될 겁니다.

```
>>> query = 'puppy'
>>> key_value_func(query)
3.2 # = 0.9 * 1 + 0.1 * 2 + 0.7 * 3
```

무슨 의미인지는 모르겠지만 ***3.2***라는 값이 나왔습니다. 

- 만약, ***dic***의 ***value***에는 100차원의 voctor로 들어있었다면 어떻게 될까요? 
- 거기에, ***query***와 ***key***값 모두 vector라면 어떻게 될까요? 즉, Word Embedding Vector라면?
- 그리고, ***dic***의 ***key***값과 ***value***값이 서로 같다면 어떻게 될까요?

### Linear Transform



## 설명

## 코드
