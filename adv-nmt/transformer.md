# 트랜스포머(Transformer, Attention is All You Need)

페이스북에서 CNN을 활용한 번역기에 대한 논문 <comment> [[Gehring et al.2017]](https://arxiv.org/pdf/1705.03122.pdf) </comment>을 내며, 기존의 구글 신경망 번역기보다 속도나 성능면에서 뛰어남을 자랑하자, 이에 질세라 구글에서 바로 곧이어 발표한 [Attention is all you need [Vaswani at el.2017]](https://arxiv.org/pdf/1706.03762.pdf) 논문입니다. 실제로 아카이브(ArXiv)에 페이스북이 5월에 해당 논문을 발표한데 이어서 6월에 이 논문이 발표되었습니다. 이 논문에서 구글은 아직까지 번역에 있어서 자신들의 기술력 우위성을 주장하였습니다.

논문의 제목에서 알 수 있듯이, 이 방법은 기존의 어텐션 연산만을 활용하여 sequence-to-sequence를 구현하였고, 성능과 속도 두마리 토끼를 성공적으로 잡아냈습니다. 하지만 이 모델은 기존의 어텐션을 활용한 것이기 때문에, 어텐션에 대한 정확한 이해가 있다면 따라가는데 크게 어려움이 없습니다.

## 구조

![트랜스포머의 구조](../assets/nmt-transformer-1.png)

"Attention is all you need"라는 제목답게 이 구조는 정말로 어텐선만 사용하여 인코딩과 디코딩을 전부 수행합니다. 그리고 저자는 이 모델 구조를 트랜스포머(transformer)라고 이름 붙였습니다. 트랜스포머의 인코더와 디코더를 이루고 있는 서브모듈(sub-module)은 크게 3가지로 나뉘어 집니다.

|명칭|역할|
|-|-|
|셀프 어텐션|이전 레이어의 출력에 대해서 어텐션 연산을 수행합니다.|
|어텐션|기존의 sequence-to-sequence와 같이 인코더의 결과에 대해서 어텐션 연산을 수행 합니다.|
|피드포워드 레이어|어텐션 레이어를 거쳐 얻은 결과물을 최종적으로 정리합니다.|

인코더는 다수의 셀프 어텐션 레이어와 피드포워드 레이어로 이루어져 있습니다. 디코더는 다수의 셀프 어텐션과 일반 어텐션이 번갈아 나타나고 피드포워드 레이어가 나타납니다.

### 포지션 임베딩 (Position Embedding)

RNN은 데이터를 순차적으로 받으면서 자동적으로 순서에 대한 정보를 기록하게 됩니다. 하지만 트랜스포머는 RNN을 이용하지 않기 때문에, 순서 정보를 단어와 함께 주는 것이 필요합니다. 왜냐하면 같은 단어라 하더라도 위치에 따라서 그 쓰임새와 역할 의미가 달라질 수 있기 때문입니다. 따라서 포지션 임베딩이라는 방법을 통해서 위치 정보를 나타내고자 하였습니다.

![포지션 임베딩의 직관적인 설명](image_needed)

$$\begin{gathered}
\text{PE}(\text{pos}, 2i) = \sin(\text{pos} / 10000^{2i / d_{model}}) \\
\text{PE}(\text{pos}, 2i + 1) = \cos(\text{pos} / 10000^{2i / d_{model}})
\end{gathered}$$

포지션 임베딩의 결과값의 차원은 단어 임베딩 벡터의 차원과 같으며, 두 벡터를 더하여 인코더 또는 디코더의 입력으로 넘겨주게 됩니다.

### 어텐션

![트랜스포머의 어텐션 구성](../assets/nmt-transformer-2.png)

트랜스포머의 어텐션 방식은 여러 개의 어텐션으로 구성된 멀티헤드(multi-head) 어텐션을 제안합니다. 이는 마치 CNN에서 여러 개의 필터(커널)가 다양한 피쳐들를 뽑아 내는 것과 같은 원리라고 볼 수 있습니다. 이전 챕터에서 어텐션에 대해서 설명 할 때, 어텐션은 쿼리(query)를 만들기 위한 선형 변환을 배우는 과정이라고 하였습니다. 이때, 다양한 쿼리들을 만들어내어 다양한 정보들을 추출할 수 있다면 훨씬 더 유용할 것 입니다. 따라서 멀티헤드를 통해 어텐션 여럿을 동시에 수행합니다.

기본적인 어텐션의 수식은 아래와 같습니다. 기본적인 어텐션은 원래 그냥 행렬 곱 연산인데, 'scaled'라는 이름이 붙은 이유는 키(key)의 차원 $d_k$ 이 주어졌을 때, 행렬 곱 연산 결과값을 $\sqrt{d_k}$로 나누어주었기 때문입니다. 이 나눗셈을 통해 좀 더 안정적인 학습 결과를 얻을 수 있다고 논문에서는 밝히고 있습니다. 이외에는 이전 챕터에서 다루었던 어텐션과 동일합니다.

아래와 같이 우리는 쿼리(query), 키(key), 밸류(value)를 입력으로 받는 어텐션 함수를 정의할 수 있습니다. 스칼라(scalar) 값으로 행렬 곱의 결과를 나누어 주는 것을 제외하면 이전에 배운 내용과 동일함을 알 수 있습니다.

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

위의 어텐션 함수를 활용하여 아래의 멀티헤드 함수를 따라가보도록 하겠습니다.

$$\begin{gathered}
\text{MultiHead}(Q,K,V)=[head_1;head_2;\cdots;head_h]W^O \\
\text{where }head_i=\text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\end{gathered}$$

여기서 Q, K, V는 쿼리와 키, 밸류를 의미 합니다. 셀프 어텐션의 경우에는 Q, K, V 모두 같은 값으로써, 이전 레이어의 결과를 받아오게 됩니다. 그리고 일반 어텐션의 경우에는 쿼리 Q는 이전 레이어의 결과가 되고, K와 V는 인코더의 마지막 레이어 결과가 됩니다. 이 경우 Q, K, V 텐서의 크기는 아래와 같습니다. 여기서 n은 소스(source) 문장의 길이, m은 타겟(target) 문장의 길이를 의미 합니다.

$$\begin{gathered}
|Q|=(\text{batch\_size},m,\text{hidden\_size}) \\
|K|=|V|=(\text{batch\_size},m,\text{hidden\_size}) \\
\text{where }n\text{ is length of source sentence, and }m\text{ is length of target sentence.}
\end{gathered}$$

그리고 멀티헤드 함수에는 선형변환을 위한 수많은 뉴럴 네트워크 웨이트 파라미터 $W_i^Q, W_i^K, W_i^V \text{ and } W^O$ 들이 존재하는 것을 볼 수 있습니다. 이들의 크기는 아래와 같습니다.

$$\begin{gathered}
|W_i^Q|=|W_i^K|=|W_i^V|=(\text{hidden\_size},\text{head\_size}) \\
|W^O|=(\text{head\_size}\times{h},\text{hidden\_size}) \\
\\
\text{where }\text{hidden\_size}=512,h=8\text{ and }\text{hidden\_size}=\text{head\_size}\times{h}
\end{gathered}$$

위와 같이 구글은 하이퍼 파라미터인 hidden_size와 head의 갯수를 논문에 제시하였습니다. 좀 더 자세한 셋팅 값은 논문을 참고하기 바랍니다.

### Self Attention for Decoder

Decoder의 self-attention은 encoder의 그것과 조금 다릅니다. 이전 레이어의 출력값을 가지고 Q, K, V를 구성하는 것은 같지만, 약간의 제약이 더해졌습니다. 그 이유는 inference 할 때, 다음 time-step의 값을 알 수 없기 때문입니다. 따라서, self-attention을 하더라도 이전 time-step에 대해서만 접근이 가능하도록 해야 합니다. 이를 구현하기 위해서 scaled dot-product attention 계산을 할 때에 masking을 추가하여, 미래의 time-step에 대해서는 weight를 가질 수 없도록 하였습니다.

### Position-wise Feed Forward Layer

$$\text{FFN}(x) = \max{(0, xW_1 + b_1)}W_2 + b_2\text{ where }d_{ff} = 2048$$

사실 여기에서 소개한 이 layer는 기존의 fully connected feed forward layer라기보단, kernel size가 1인 convolutional layer라고 볼 수 있습니다. Channel숫자가 512 $\rightarrow$ 2048 으로 가는 convolution과, 2048 $\rightarrow$ 512로 가는 convolution으로 이루어져 있는 것 입니다.

## Evaluation

![Transformer의 성능 비교](../assets/nmt-transformer-3.png)

Google은 transformer를 통해서 State of the Art의 성능을 달성했다고 보고하였습니다. 뿐만아니라, 기존의 RNN 및 Facebook의 ConvS2S보다 훨씬 빠른 속도로 훈련이 가능하다고 하였습니다. 실제로 위의 table을 보면, transformer의 training cost의 magnitude는 $10^{18}$ 으로, 대부분의 다른 방식 $10^{19}$ 와 급격한 차이를 보이는 것을 알 수 있습니다.

또 하나의 속도 개선의 원인은 input feeding의 부재입니다. RNN기반의 방식은 input feeding이 도입되면서 decoder를 훈련할 때 모든 time-step을 한번에 할 수 없게 되었습니다. 이로 인해서 대부분의 병목이 decoder에서 발생합니다. 하지만 transformer는 input feeding이 없기 때문에 한번에 모든 time-step에 대해서 계산할 수 있게 되었습니다.

비록 transformer가 최고 성능을 달성하긴 헀지만 그 모델 구조의 과격함 때문인지 (Facebook의 모델과 함께) 아직 주류로 편입되지 않았습니다. 아직 대부분의 논문들은 이 구조를 비교대상으로 논하기보다, RNN구조의 seq2seq를 대상으로 실험을 비교/진행 하곤 합니다.

## 결론