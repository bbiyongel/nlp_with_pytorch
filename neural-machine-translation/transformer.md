# Transformer

Facebook에서 CNN을 활용한 번역기에 대한 논문을 내며, 기존의 GNMT 보다 속도나 성능면에서 뛰어남을 자랑하자, 이에 질세라 Google에서 바로 곧이어 발표한 [Attention is all you need \[Vaswani at el.2017\]](https://arxiv.org/pdf/1706.03762.pdf) 논문입니다. 실제로 ArXiv에 Facebook이 5월에 해당 논문을 발표한데 이어서 6월에 이 논문이 발표되었습니다. 이 논문을 한문장으로 요약하면 **"그래도 아직 우리가 더 잘하지롱"** 정도가 되겠습니다. 덕분에 NMT 기술이 덩달아 발전하는 순기능까지 있었고, 개인적으로는 아주 재미있는 구경이었습니다. 

![](/assets/nmt-transformer-1.png)

## Position Embedding

$$
PE(pos, 2i) = \sin(pos / 10000^{2i / d_{model}})
$$
$$
PE(pos, 2i + 1) = \cos(pos / 10000^{2i / d_{model}})
$$

## Attention

![](/assets/nmt-transformer-2.png)

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
$$
MultiHead(Q, K, V) = [head_1;head_2;\cdots;head_h]W^O
$$
$$
where~head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$
$$
where~W_i^Q \in \mathbb{R}^{d_{model}\times d_k}, W_i^K \in \mathbb{R}^{d_{model}\times d_k}, W_i^V \in \mathbb{R}^{d_{model}\times d_v}~and~W^O \in \mathbb{R}^{hd_{v}\times d_{model}}
$$
$$
d_k = d_v = d_{model}/h = 64
$$
$$
h = 8, d_{model} = 512
$$
$$
FFN(x) = \max{(0, xW_1 + b_1)}W_2 + b_2
$$
$$
d_{ff} = 2048
$$
