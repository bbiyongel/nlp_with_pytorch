# Transformer (Attention is All You Need)

Facebook에서 CNN을 활용한 번역기에 대한 논문을 내며, 기존의 GNMT 보다 속도나 성능면에서 뛰어남을 자랑하자, 이에 질세라 Google에서 바로 곧이어 발표한 [Attention is all you need [Vaswani at el.2017]](https://arxiv.org/pdf/1706.03762.pdf) 논문입니다. 실제로 ArXiv에 Facebook이 5월에 해당 논문을 발표한데 이어서 6월에 이 논문이 발표되었습니다. 이 논문에서 Google은 아직까지 번역에 있어서 자신들의 기술력 우위성을 주장하였습니다.

## Architecture

![Transformer 아키텍처](../assets/nmt-transformer-1.png)

"Attention is all you need"라는 제목의 논문답게 이 논문은 정말로 Attention만 구현해서 모든것을 해냅니다. 그리고 저자는 이 모델 구조를 Transformer라고 이름 붙였습니다. Encoder와 decoder를 설명하기에 앞서, sub-module부터 소개하겠습니다. Encoder와 decoder를 이루고 있는 sub-module은 크게 3가지로 나뉘어 집니다.

|명칭|역할|
|-|-|
|Self-attention|이전 layer의 output에 대해서 attention을 수행합니다.|
|Attention|Encoder의 output에 대해서 기존의 seq2seq와 같이 attention을 수행합니다.|
|Feed Forward Layer|attention layer을 거쳐 얻은 결과물을 최종적으로 정리합니다.|

Encoder는 다수의 self-attention layer와 feed forward layer로 이루어져 있습니다. Decoder는 다수의 self-attention과 attention이 번갈아 나타나고, feed forward layer가 있습니다. 이처럼 Transformer는 구성되며 각 모듈에 대한 자세한 설명은 아래와 같습니다.

### Position Embedding

이전 Facebook 논문과 마찬가지로, RNN을 이용하지 않기 때문에, 위치정보를 단어와 함께 주는 것이 필요합니다. 따라서 Google에서도 마찬가지로 position embedding을 통해서 위치 정보를 나타내고자 하였으며, 그 수식은 약간 다릅니다.

$$\begin{gathered}
\text{PE}(\text{pos}, 2i) = \sin(\text{pos} / 10000^{2i / d_{model}}) \\
\text{PE}(\text{pos}, 2i + 1) = \cos(\text{pos} / 10000^{2i / d_{model}})
\end{gathered}$$

Position embedding의 결과값의 dimension은 word embedding의 dimension과 같으며, 두 값을 더하여 encoder 또는 decoder의 입력으로 넘겨주게 됩니다.

### Attention

![Transformer의 Attention 구성](../assets/nmt-transformer-2.png)

이 논문에서의 Attention방식은 여러개의 attention으로 구성된 multi-head attention을 제안합니다. 마치 Convolution layer에서 여러개의 filter가 있어서 여러가지 다양한 feature를 뽑아 내는 것과 같은 원리라고 볼 수 있습니다.

기본적인 attention의 수식은 아래와 같습니다. 기본적인 attention은 원래 그냥 dot-product attention인데 scaled라는 이름이 붙은 이유는 key의 dimension인 $\sqrt{d_k}$로 나누어주었기 때문입니다. 이외에는 이전 섹션에서 다루었던 attention과 같습니다.

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

이렇게 구성된 attention을 하나의 head로 삼아 Multi-Head Attention을 구성합니다.

$$\begin{aligned}
\text{MultiHead}(Q, K, V) &= [head_1;head_2;\cdots;head_h]W^O \\
\text{where }head_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{where }W_i^Q &\in \mathbb{R}^{d_{model}\times d_k}, W_i^K \in \mathbb{R}^{d_{model}\times d_k}, \\
W_i^V &\in \mathbb{R}^{d_{model}\times d_v}\text{ and }W^O \in \mathbb{R}^{hd_{v}\times d_{model}} \\ \\
d_k = d_v &= d_{model}/h = 64 \\
h &= 8\text{ and }d_{model} = 512 \\
\end{aligned}$$

이때에 각 head의 Q, K, V 마다 다른 W를 곱해줌으로써 각각 linear transformation형태를 취해 줍니다. 즉, head마다 필요한 다른 정보(feature)를 attention을 통해 encoding 할 수 있게 됩니다. 해당 논문에서는 hidden size를 512로 하고 이를 8개의 head로 나누어 각 head의 hidden size는 64가 되도록 하였습니다.

실제 구현을 할 때에는 self attention의 경우에는 이전 layer의 출력값이 모두 Q, K, V를 이루게 됩니다. 같은 값이 Q, K, V로 들어가지만 linear transform을 해주기 때문에 상관이 없습니다. Decoder에서 수행하는 encoder에 대한 attention을 할 때에는, Q는 decoder의 이전 layer의 출력값이 되지만, K, V는 encoder의 출력값이 됩니다.

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

## 읽을거리

