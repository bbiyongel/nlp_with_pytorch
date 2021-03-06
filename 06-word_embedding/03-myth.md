# 흔한 오해

우리는 이번 챕터를 통해 단어를 벡터로 표현하는 방법에 대해서 살펴보고 있습니다. 이어지는 섹션에서 skip-gram 또는 GloVe를 사용하여 One-hot 인코딩의 희소(sparse) 벡터를 차원 축소(dimension reduction)하여 훨씬 작은 차원의 덴스(dense) 벡터로 표현하는 방법에 대해 다룰 것 입니다.

이에 앞서, 하지만 많은 분들이 헷갈려 하는 부분이 있다면, 이렇게 훈련한 단어 임베딩 벡터를 추후 우리가 다룰 텍스트 분류, 언어모델, 번역 등의 딥러닝 모델들의 입력으로 사용할 것이라고 생각한다는 점 입니다. 이때 이 임베딩 벡터를 pretrained 임베딩 벡터라고 부릅니다. 비록 Word2Vec을 통해 얻은 단어 임베딩 벡터는 훌륭하게 단어의 특성을 잘 반영하고 있지만, 텍스트 분류, 언어모델, 번역의 문제 해결을 위한 최적의 벡터 임베딩이라고는 볼 수 없습니다. 다시 말하면 텍스트 분류 또는 기계번역을 위한 목적함수(objective function)은 분명히 Word2Vec과 다른 형태로 존재합니다. 따라서 다른 목적함수를 통해 훈련한 임베딩 벡터는 원래의 목적에 맞지 않을 가능성이 높습니다.

예를 들어 긍정/부정 감성 분류를 위한 텍스트 분류 문제의 경우에는 '행복'이라는 단어가 매우 중요한 특징(feature)가 될 것이고, 이를 표현하기 위한 임베딩 벡터가 존재할 것 입니다. 하지만 기게 번역 문제에서는 '행복'이라는 단어는 그냥 일반적인 단어에 지나지 않을 것 입니다., 이 분류 문제를 위한 '행복' 단어의 임베딩 벡터의 값은 이전 긍정/부정 분류 문제의 값과 당연히 달라지게 될 것 입니다. 따라서 문제의 특성을 고려하지 않은 단어 임베딩 벡터는 그다지 좋은 방법이 될 수 없습니다.

## 정석: Word2Vec을 사용하지 않고 뉴럴 네트워크를 훈련 시키는 방법

우리는 Word2Vec을 사용하여 단어를 저차원의 임베딩 벡터로 변환하지 않더라도, 문제의 특성에 맞는 단어 임베딩 벡터를 구할 수 있습니다. 파이토치를 비롯한 여러 딥러닝 프레임워크는 'Embedding Layer'라는 레이어 아키텍처를 제공합니다. 이 레이어는 아래와 같이 bias가 없는 'Linear Layer'와 같은 형태를 갖고 있습니다.

$$\begin{gathered}
y=\text{emb}(x)=Wx, \\
\text{where }W\in\mathbb{R}^{d\times|V|}\text{ and }|V|\text{ is size of vocabulary}.
\end{gathered}$$

쉽게 생각하면 $W$ 는 $d\times|V|$ 크기의 2차원의 행렬 입니다. 따라서 입력으로 one-hot 벡터가 주어지게 되면, $W$ 의 특정 컬럼(column) <comment> 구현 방법에 따라 또는 row 만 반환하게 됩니다. </comment>

![임베딩 레이어의 동작 개념](../assets/06-03-01.png)

따라서 최종적으로 모델으로부터 구한 손실값(loss)에 따라 back-propagation 및 그래디언트 디센트(gradient descent)를 수행하게 되면, 자동적으로 임베딩 레이어의 웨이트(weight) $W$ 의 값을 구할 수 있게 될 것 입니다.

물론 실제 구현에 있어서는 이렇게 큰 임베딩 레이어 웨이트와 one-hot 인코딩 벡터를 곱하는 것은 매우 비효율적이므로, 단순히 테이블에서 검색(lookup)하는 작업을 수행 합니다. 따라서 우리는 단어를 나타냄에 있어 (임베딩 레이어의 입력으로) one-hot 벡터를 굳이 넘겨줄 필요 없이, 1이 존재하는 단어의 인덱스(index) 정수 값만 입력으로 넘겨주면 임베딩 벡터를 얻을 수 있습니다. 또한 굳이 임베딩 레이어와의 계산의 효율성이 아니더라도, one-hot 인코딩 벡터를 표현함에 있어서 1이 존재하는 단어의 인덱스 정수 값만 가지고 있으면 되는 것은 자명 합니다. <comment> n차원의 one-hot 인코딩 벡터에서 1은 한 개이며, $n-1$ 개의 0으로 채워져 있습니다. </comment>

추후 실제 앞으로 우리가 다룰 텍스트 분류나 기계번역 챕터에서 구현된 것을 살펴보면, Word2Vec을 사용하여 단어를 임베딩 벡터로 변환한 후 뉴럴 네트워크에 직접 넣어주는 것이 아닌, 위에서 언급한대로 임베딩 레이어를 사용하여 one-hot 인코딩 벡터를 입력으로 넣어주도록 구현한 것을 알 수 있습니다.

## 그래도 적용해 볼만한 경우

그래도 사전훈련된 단어 임베딩 벡터를 적용한 훈련을 고려해 볼 만한 몇 가지 상황이 있습니다. 예를 들어 준비 된 코퍼스(corpus)의 양이 너무 적고, 이 때 외부로부터 많은 양의 말뭉치를 통해 미리 훈련한 워드 임베딩 벡터를 구할 수 있는 특수한 경우를 생각 해 볼 수 있습니다.

하지만 기본 정석 방법대로 먼저 베이스라인(baseline) 모델을 만든 후에, 성능을 끌어올리기 위한 여러가지 방법들을 시도 할 때, 사전훈련된 단어 임베딩 벡터의 사용을 고려해 볼 수도 있습니다.

또한 우리가 마지막 장인 전이학습(transfer learning)에서 다룰 고도화된 언어모델(langauge model)을 사용하여 사전훈련하여 접근 해 볼 수 있습니다. 우리는 이와 관련된 내용을 이 책의 마지막 장에서 다루도록 하겠습니다.
