# Transformer

## 소개

[\[Vaswani at el.2017\]](https://arxiv.org/pdf/1706.03762.pdf)

![](/assets/nmt-transformer-1.png)

### Position Embedding

$$
PE(pos, 2i) = \sin(pos / 10000^{2i / d_{model}})
$$
$$
PE(pos, 2i + 1) = \cos(pos / 10000^{2i / d_{model}})
$$

### Self Attention

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
where~W_i^Q \in ℝ^{d_{model}×d_k}, W_i^K \in ℝ^{d_{model}×d_k}, W_i^V \in ℝ^{d_{model}×d_v}~and~W^O \in ℝ^{hd_{v}×d_{model}}
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

### Attention

## 구조 설계

## 설명



