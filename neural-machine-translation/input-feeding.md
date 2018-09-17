# Input Feeding

## Overview

Decoder output과 Attention 결과값을 concatenate한 이후에 Generator 모듈에서 softmax를 취하여 $\hat{y}_{t}$을 구합니다. 하지만 이러한 softmax 과정에서 많은 정보(예를 들어 attention 정보 등)가 손실됩니다. 따라서 단순히 다음 time-step에 $\hat{y}_{t}$을 feeding 하는 것보다, concatenation layer의 출력도 같이 feeding 해주면 정보의 손실 없이 더 좋은 효과를 얻을 수 있습니다.

![Input Feeding이 추가 된 Sequence-to-Sequence 아키텍처](../assets/nmt-seq2seq-with-attention-and-input-feeding-architecture.png)

$y$와 달리 concatenation layer의 출력은 $y$가 embedding layer에서 dense vector(=embedding vector)로 변환되고 난 이후에 embedding vector와 concatenate되어 decoder RNN에 입력으로 주어지게 됩니다. 이러한 과정을 input feeding이라고 합니다.

$$
\begin{gathered}
h_{t}^{src} = \text{RNN}_{enc}(\text{emb}_{src}(x_t), h_{t-1}^{src}) \\
H^{src} = [h_{1}^{src}; h_{2}^{src}; \cdots; h_{n}^{src}] \\
h_{t}^{tgt} = \text{RNN}_{dec}([\text{emb}_{tgt}(y_{t-1});\tilde{h}_{t-1}^{tgt}], h_{t-1}^{tgt})\text{ where}h_{0}^{tgt}=h_{n}^{src}\text{ and }y_{0}=BOS. \\
w=\text{softmax}({h_{t}^{tgt}}^T W \cdot H^{src}) \\
c = H^{src} \cdot w\text{ and }c\text{ is a context vector}. \\
\tilde{h}_{t}^{tgt}=\tanh(\text{linear}_{2hs \rightarrow hs}([h_{t}^{tgt}; c])) \\
\hat{y}_{t}=\text{softmax}(\text{linear}_{hs \rightarrow |V_{tgt}|}(\tilde{h}_{t}^{tgt})) \\
\text{where }hs\text{ is hidden size of RNN, and }|V_{tgt}|\text{ is size of output vocabulary}.
\end{gathered}
$$

위의 수식은 attention과 input feeding이 추가된 seq2seq의 처음부터 끝까지 입니다. $\text{RNN}_{dec}$는 이제 $\tilde{h}_{t-1}^{tgt}$를 입력으로 받기 때문에, 모든 time-step을 한번에 처리하도록 구현할 수 없다는 점이 구현상의 차이점입니다.

## Disadvantage

이 방식은 훈련 속도 저하라는 단점을 가집니다. input feeding이전 방식에서는 훈련 할 때에는 teacher forcing 방식이기 때문에(모든 입력을 알고 있기 때문에), encoder와 마찬가지로 decoder도 모든 time-step에 대해서 한번에 feed-forward 작업이 가능했습니다. 하지만 input feeding으로 인해, decoder RNN의 input으로 이전 time-step의 결과가 필요하게 되어, 다시 추론(inference)할 때 처럼 auto-regressive 속성으로 인해 각 time-step 별로 순차적으로 계산을 해야 합니다.

하지만 이 단점이 크게 부각되지 않는 이유는 어차피 추론(inference)단계에서는 decoder는 input feeding이 아니더라도 time-step 별로 순차적으로 계산되어야 하기 때문입니다. 추론 단계에서는 이전 time-step의 output인 $\hat{y}_t$를 decoder(정확하게는 decoder 이전의 embedding layer)의 입력으로 사용해야 하기 때문에, 어쩔 수 없이 병렬처리가 아닌 순차적으로 계산해야 합니다. 따라서 추론 할 때, input feeding으로 인한 속도 저하는 거의 없습니다.

## Evaluation

| NMT system | Perplexity | En/De BLEU |
| --- | --- | --- |
| Base | 10.6 | 11.3 |
| Base + reverse | 9.9 | 12.6(+1.3) |
| Base + reverse + dropout | 8.1 | 14.0(+1.4) |
| Base + reverse + dropout + attention | 7.3 | 16.8(+2.8) |
| Base + reverse + dropout + attention + feed input | 6.4 | 18.1(+1.3) |

현재 방식을 처음 제안한 [\[Loung et al.2015\] Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)에서는 실험 결과를 위와 같이 주장하였습니다. 실험 대상은 아래와 같습니다.

* Baseline: 기본적인 seq2seq 모델
* Reverse: Bi-directional LSTM을 encoder에 적용
* Dropout: probability 0.2
* Global Attention
* Input Feeding

우리는 이 실험에서 attention과 input feeding을 사용함으로써, 훨씬 더 나은 성능을 얻을 수 있음을 알 수 있습니다.

## Code

NMT를 목표로하는 seq2seq를 PyTorch로 구현하는 방법을 소개합니다. 이번 챕터에서 사용될 전체 코드는 저자의 깃허브에서 다운로드 할 수 있습니다. (업데이트 여부에 따라 코드가 약간 다를 수 있습니다.)

- github repo url: https://github.com/kh-kim/simple-nmt

### Decoder Class

```python
class Decoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers = 4, dropout_p = .2):
        super(Decoder, self).__init__()

        # Be aware of value of 'batch_first' parameter and 'bidirectional' parameter.
        self.rnn = nn.LSTM(word_vec_dim + hidden_size, hidden_size, num_layers = n_layers, dropout = dropout_p, bidirectional = False, batch_first = True)

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (batch_size, 1, word_vec_dim)
        # |h_t_1_tilde| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            # If this is the first time-step,
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        # Input feeding trick.
        x = torch.cat([emb_t, h_t_1_tilde], dim = -1)

        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1)

        return y, h
```

### Sequence-to-Sequence

#### Initialization

```python
    def __init__(self, input_size, word_vec_dim, hidden_size, output_size, n_layers = 4, dropout_p = .2):
        self.input_size = input_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_dim)
        self.emb_dec = nn.Embedding(output_size, word_vec_dim)
        
        self.encoder = Encoder(word_vec_dim, hidden_size, n_layers = n_layers, dropout_p = dropout_p)
        self.decoder = Decoder(word_vec_dim, hidden_size, n_layers = n_layers, dropout_p = dropout_p)
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)
```

#### Mask Generation

```python
    def generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(), x.new_ones(1, (max_length - l))], dim = -1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim = 0).byte()

        return mask
```

#### Convert Hidden State from Encoder to Decoder

![](../assets/nmt-encoder-to-decoder.png)

```python
    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens

        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim = -1)]
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim = -1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)
```

#### Forward

```python
    def forward(self, src, tgt):
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Get word embedding vectors for every time-step of input sentence.
        emb_src = self.emb_src(x)
        # |emb_src| = (batch_size, length, word_vec_dim)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        # Merge bidirectional to uni-directional
        # We need to convert size from (n_layers * 2, batch_size, hidden_size / 2) to (n_layers, batch_size, hidden_size).
        # Thus, the converting operation will not working with just 'view' method.
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        # You can use 'merge_encoder_hiddens' method, instead of using above 3 lines.
        # 'merge_encoder_hiddens' method works with non-parallel way.
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        h_0_tgt = (h_0_tgt, c_0_tgt)

        emb_tgt = self.emb_dec(tgt)
        # |emb_tgt| = (batch_size, length, word_vec_dim)
        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt
        # Run decoder until the end of the time-step.
        for t in range(tgt.size(1)):
            # Teacher Forcing: take each input from training set, not from the last time-step's output.
            # Because of Teacher Forcing, training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder, this causes severe bottle-neck.
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            # |emb_t| = (batch_size, 1, word_vec_dim)
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            # |decoder_output| = (batch_size, 1, hidden_size)
            # |decoder_hidden| = (n_layers, batch_size, hidden_size)

            context_vector = self.attn(h_src, decoder_output, mask)
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output, context_vector], dim = -1)))
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim = 1)
        # |h_tilde| = (batch_size, length, hidden_size)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (batch_size, length, output_size)

        return y_hat
```

#### Seq2Seq Class

```python
class Seq2Seq(nn.Module):
    
    def __init__(self, input_size, word_vec_dim, hidden_size, output_size, n_layers = 4, dropout_p = .2):
        self.input_size = input_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_dim)
        self.emb_dec = nn.Embedding(output_size, word_vec_dim)
        
        self.encoder = Encoder(word_vec_dim, hidden_size, n_layers = n_layers, dropout_p = dropout_p)
        self.decoder = Decoder(word_vec_dim, hidden_size, n_layers = n_layers, dropout_p = dropout_p)
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(), x.new_ones(1, (max_length - l))], dim = -1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim = 0).byte()

        return mask

    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens

        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim = -1)]
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim = -1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)

    def forward(self, src, tgt):
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Get word embedding vectors for every time-step of input sentence.
        emb_src = self.emb_src(x)
        # |emb_src| = (batch_size, length, word_vec_dim)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        # Merge bidirectional to uni-directional
        # We need to convert size from (n_layers * 2, batch_size, hidden_size / 2) to (n_layers, batch_size, hidden_size).
        # Thus, the converting operation will not working with just 'view' method.
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        # You can use 'merge_encoder_hiddens' method, instead of using above 3 lines.
        # 'merge_encoder_hiddens' method works with non-parallel way.
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        h_0_tgt = (h_0_tgt, c_0_tgt)

        emb_tgt = self.emb_dec(tgt)
        # |emb_tgt| = (batch_size, length, word_vec_dim)
        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt
        # Run decoder until the end of the time-step.
        for t in range(tgt.size(1)):
            # Teacher Forcing: take each input from training set, not from the last time-step's output.
            # Because of Teacher Forcing, training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder, this causes severe bottle-neck.
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            # |emb_t| = (batch_size, 1, word_vec_dim)
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            # |decoder_output| = (batch_size, 1, hidden_size)
            # |decoder_hidden| = (n_layers, batch_size, hidden_size)

            context_vector = self.attn(h_src, decoder_output, mask)
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output, context_vector], dim = -1)))
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim = 1)
        # |h_tilde| = (batch_size, length, hidden_size)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (batch_size, length, output_size)

        return y_hat
```