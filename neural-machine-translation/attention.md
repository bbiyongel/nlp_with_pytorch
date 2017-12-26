# Attention

## ![](/assets/seq2seq_with_attn_architecture.png)

## 소개

### Key-Value function

Attention을 본격 소개하기 전에 먼저 우리가 알고 있는 자료형을 짚고 넘어갈까 합니다. Key-Value 또는 [Python에서 Dictionary](https://wikidocs.net/16)라고 부르는 자료형 입니다.

```
dic = {'dog': 0, 'computer': 1, 'car': 2}
```

위와 같이 ***Key***와 ***Value***에 해당하는 값들을 넣고 Key를 통해 Value 값에 접근 할 수 있습니다. 위의 작업을 함수로 나타낸다면, 아래와 같이 표현할 수 있을겁니다. (물론 실제 Python Dictionary 동작은 매우 다릅니다.)

```
def key_value_func(query):
    weights = []
    
    for key in dic.keys():
        if key == query:
            weights += [1.]
        else:
            weights += [.0]
    
    answer = 0
    
    for weight, value in zip(weights, dic.values()):
        answer += weight * value
        
    return answer
```

코드를 살펴보면, 순차적으로 ***dic*** 내부의 key값들과 ***query*** 값을 비교하며, 같은 key가 있을 경우 weights에 

### Query-Key-Value function

## 설명

## 코드
