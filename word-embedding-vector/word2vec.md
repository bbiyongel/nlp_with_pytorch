# Word2Vec

2011년 Tomas Mikolov는 Word2Vec이라는 방법을 제시하여 학계에 큰 파장을 일으켰습니다. 물론 이전부터 Neural Network를 통해 단어를 임베딩하고자 하는 시도는 많았지만, Mikolov는 복잡한 신경망 네트워크를 사용하여 word embedding 벡터를 계산하는데 의문을 가졌습니다. 이에 빠르고 쉽고 효율적으로 임베딩하는 word2vec을 통해, 딥러닝 연구자들의 자연어처리에 대한 이해도를 한단계 끌어올렸습니다.

Word2Vec은 단어를 임베딩 하는 방법을 2가지 제시하였습니다. 두 방법 모두 공통된 가정을 갖고 있습니다. 두 방법에 사용된 가정은 함께 나타나는 단어가 비슷할 수록 비슷한 벡터 값을 가질 것이라는 것 입니다.

![](../assets/intro-word2vec.png)

위와 같이 두 방법 모두 윈도우(window)의 크기가 주어지면, 특정 단어를 기준으로 윈도우 내의 주변 단어들을 사용하여 단어 임베딩을 학습합니다. 단, 윈도우 내에서의 위치는 고려되지 않습니다. 하지만 이때 단어의 위치 정보가 무시되는 것은 아닙니다. 윈도우 자체가 단어의 위치 정보를 내포하고 있기 때문입니다. 문장 내 단어의 위치에 따라서 윈도우에 포함되는 단어가 달라질 것이기 때문입니다.

## CBOW & Skip-gram

CBOW(Continuous Bag of Words)는 주변에 나타나는 단어를 입력으로 주어 해당 단어를 예측하도록 하는 신경망 구조를 가진 모델을 통해 단어 임베딩을 학습합니다.

Skip-gram은 단어를 입력으로 주어 주변에 나타나는 단어를 예측하도록 하는 신경망 구조를 가진 모델을 통해 단어 임베딩을 학습합니다.

보통 Skip-gram이 CBOW보다 성능이 뛰어난 것으로 알려져 있고, 따라서 좀 더 널리 쓰입니다.

## Architecture Detail

Skip-gram의 구조를 좀 더 자세하게 살펴보도록 하겠습니다. 먼저 skip-gram을 학습하는 과정은 아래와 같습니다. 파라미터 $\theta$는 Maximum Likelihood Estimation (MLE)를 통해 아래의 수식을 최대로 하는, $w_t$가 주어졌을떄, 앞뒤 $n$개의 단어를 예측하도록 훈련 됩니다.

$$
\hat{\theta}=\underset{\theta}{\text{argmax}}\sum_{t=1}^T{\Big(\sum_{i=1}^n{\log{P(w_{t-i}|w_t;\theta)}}+\sum_{i=1}^n{\log{P(w_{t+i}|w_t;\theta)}}\Big)}
$$

우리는 이전 섹션에서 embedding layer를 통해서 one-hot 인코딩 벡터를 dense한 벡터인 word embedding vector로 변환하는 방법에 대해서 다루었습니다. Skip-gram에서도 마찬가지 방법을 사용 합니다.

$$
\begin{gathered}
\hat{y}=\underset{y\in\mathcal{Y}}{\text{argmax }}\text{softmax}(W'Wx) \\
\text{where }W'\in\mathbb{R}^{|V|\times d}, W\in\mathbb{R}^{d\times|V|}\text{ and }x\in\mathbb\{0,1\}^{|V|}.
\end{gathered}
$$

위의 수식을 그림으로 표현하면 아래와 같습니다. 수식에서 볼 수 있듯이 1개의 hidden layer를 갖고 있으며, 매우 간단한 구조입니다.

이때, $W$의 각 row(그림에서는 column)가 skip-gram을 통해 얻은 단어 $x$에 대한 word embedding 벡터가 됩니다.

![Skip-gram을 통해 얻은 word embedding 벡터를 t-SNE를 통해 visualization 한 예제](../assets/intro-word-embedding.png)

<!--
## Negative Sampling

사실 위의 방법은 나름 괜찮은 방법이지만, $|V|$가 매우 클 경우에 $W$와 $W'$가 커짐으로 인해서 메모리와 연산량에 있어 부하로 작용할 수 있습니다. 따라서, 이때 negative sampling 방법을 사용하여 우리는 좀 더 효율적으로 skip-gram을 구현할 수 있습니다.
-->