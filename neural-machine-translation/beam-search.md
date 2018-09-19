# Inference

## Overview

이제까지 $X$와 $Y$가 모두 주어진 훈련상황을 가정하였습니다만, 이제부터는 $X$만 주어진 상태에서 $\hat{Y}$을 예측하는 방법에 대해서 서술하겠습니다. 이러한 과정을 우리는 추론(inference) 또는 탐색(search)이라고 부릅니다. 우리가 기본적으로 이 방식을 탐색이라고 부르는 이유는 탐색 알고리즘(search algorithm)에 기반하기 때문입니다. 결국 우리가 원하는 것은 state로 이루어진 단어(word)들 사이에서 최고의 확률을 갖는 경로(path)를 찾는 것이기 때문입니다.

## Sampling

사실 먼저 우리가 생각할 수 있는 가장 정확한 방법은 각 time-step별 $\hat{y}_t$를 고를 때, 마지막 softmax 레이어에서의 확률 분포(probability distribution)대로 랜덤 샘플링을 하는 것 입니다. 그리고 다음 time-step에서 그 선택($\hat{y}_t$)을 기반으로 다음 $\hat{y}_{t+1}$을 또 다시 샘플링하여 최종적으로 $EOS$가 나올 때 까지 샘플링을 반복하는 것 입니다. 이렇게 하면 우리가 원하는 $P(Y|X)$ 에 가장 가까운 형태의 번역이 완성될 겁니다. 하지만, 이러한 방식은 같은 입력에 대해서 매번 다른 출력 결과물을 만들어낼 수 있습니다. 따라서 우리가 원하는 형태의 결과물이 아닙니다.

## Gready Search

우리는 자료구조, 알고리즘 수업에서 DFS, BFS, Dynamic Programming 등 수많은 탐색 기법에 대해 배웠습니다. 우리는 이중에서 Greedy 알고리즘을 기반으로 탐색을 구현합니다. 즉, softmax 레이어에서 가장 값이 큰 인덱스(index)를 뽑아 해당 time-step의 $\hat{y}_t$로 사용하게 되는 것 입니다.

## Code

아래의 코드는 샘플링 또는 greedy 탐색을 위한 코드 입니다. 사실 인코더가 동작하는 부분까지는 완전히 똑같습니다. 다만, 이후 추론을 위한 부분은 기존 훈련 코드와 상이합니다. Teacher Forcing을 사용하였던 훈련 방식(실제 정답 $y_{t-1}$을 $t$ time-step의 입력으로 사용함)과 달리, 실제 이전 time-step의 출력을 현재 time-step의 입력으로 사용 합니다.

```python
    def search(self, src, is_greedy = True, max_length = 255):
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        h_0_tgt = (h_0_tgt, c_0_tgt)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        y = x.new(batch_size, 1).zero_() + data_loader.BOS
        is_undone = x.new_ones(batch_size, 1).float()
        decoder_hidden = h_0_tgt
        h_t_tilde, y_hats, indice = None, [], []
        
        # Repeat a loop while sum of 'is_undone' flag is bigger than 0, or current time-step is smaller than maximum length.
        while is_undone.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure, take the last time-step's output during the inference.
            emb_t = self.emb_dec(y)
            # |emb_t| = (batch_size, 1, word_vec_dim)

            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output, context_vector], dim = -1)))
            y_hat = self.generator(h_t_tilde)
            # |y_hat| = (batch_size, 1, output_size)
            y_hats += [y_hat]

            if is_greedy:
                y = torch.topk(y_hat, 1, dim = -1)[1].squeeze(-1)
            else:
                # Take a random sampling based on the multinoulli distribution.
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
            # Put PAD if the sample is done.
            y = y.masked_fill_((1. - is_undone).byte(), data_loader.PAD)
            is_undone = is_undone * torch.ne(y, data_loader.EOS).float()            
            # |y| = (batch_size, 1)
            # |is_undone| = (batch_size, 1)
            indice += [y]

        y_hats = torch.cat(y_hats, dim = 1)
        indice = torch.cat(indice, dim = -1)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice
```

## Beam Search

![](../assets/nmt-beam-search-concept.png)

하지만 우리는 자료구조, 알고리즘 수업에서 배웠다시피, greedy 알고리즘은 굉장히 쉽고 간편하지만, 최적의(optimal) 해를 보장하지 않습니다. 따라서 최적의 해에 가까워지기 위해서 우리는 약간의 트릭을 첨가합니다. Beam Size 만큼의 후보를 더 추적하는 것 입니다.

현재 time-step에서 Top-k개를 뽑아서 (여기서 k는 beam size와 같습니다) 다음 time-step에 대해서 k번 추론을 수행합니다. 그리고 총 $k * |V|$ 개의 softmax 결과 값 중에서 다시 top-k개를 뽑아 다음 time-step으로 넘깁니다. ($|V|$는 Vocabulary size) 여기서 중요한 점은 두가지 입니다.

$$
\begin{gathered}
\hat{y}_{t}^{k} = \underset{k\text{-th}}{\text{argmax}}\hat{Y}_t \\
\hat{Y}_{t} = f_\theta(X, y_{<t}^{1}) \cup f_\theta(X, y_{<t}^{2}) \cup \cdots \cup f_\theta(X, y_{<t}^{k}) \\
X=\{x_1, x_2, \cdots, x_n\}
\end{gathered}
$$

1. 누적 확률을 사용하여 top-k를 뽑습니다. 이때, 보통 로그 확률을 사용하므로 현재 time-step 까지의 로그확률에 대한 합을 추적하고 있어야 합니다.
2. top-k를 뽑을 때, 현재 time-step에 대해 k번 계산한 모든 결과물 중에서 뽑습니다.

Beam Search를 사용하면 좀 더 넓은 경로에 대해서 search를 수행하므로 당연히 좀 더 나은 성능을 보장합니다. 하지만, beam size만큼 번역을 더 수행해야 하기 때문에 속도에 저하가 있습니다. 다행히도 우리는 이 작업을 매 time-step마다 임시로 그때그떄 mini-batch로 만들어 수행하기 때문에, 병렬로 계산할 수 있고, 이로인해 약간의 속도 저하만 생기게 됩니다.

아래는 [Cho et al.2016]에서 주장한 beam search의 성능향상에 대한 실험 결과 입니다. 샘플링 방법은 단순한 greedy 탐색보다 더 좋은 성능을 제공하지만, beam search가 가장 좋은 성능을 보여줍니다. 특기할 점은 기계번역 문제에서는 보통 beam size를 $10$ 이하로 사용한다는 것 입니다. 

![[En-Cz: 12m training sentence pairs [Cho, arXiv 2016]](http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf)](../assets/nmt-inference-method-evaluation.png)

### How to implement

하나의 샘플에 대해서 추론을 수행하기 위한 beam search를 나타낸 그림 입니다. 여기서 beam size는 3이 되는 것을 알 수 있습니다. 보통 beam search는 $EOS$가 beam size의 크기만큼 추론 될 때까지 수행 됩니다. 즉, 아래에서는 3개 이상의 $EOS$가 출현하면 beam search는 종료 됩니다. 아래의 그림에서는 마지막 직전의 time-step에서 1개의 $EOS$가 발생한 것을 볼 수 있고, $EOS$로부터 다시 이어지는 추론 대신에, 다른 2개의 추론 중에서 top-3를 다시 선택 하는 것을 볼 수 있습니다. Top-k개를 뽑을 때는 누적 확률이 높은 순서대로 뽑게 됩니다.

![1 sample beam search (beam size=3)](../assets/nmt-single-sample-beam-search.png)

좀 더 자세하게 예를 들면, $|V|=30,000$이고 $k=3$일 때, 한 time-step의 예측 결과의 갯수는 $|V|\times k=90,000$가 됩니다. 이 9만개의 softmax 레이어 출력 유닛(vocabulary의 각 단어)들의 각각의 확률 값에 이전 time-step에서 선택된 k개의 누적 확률 값을 더해 최종 누적 확률 값 $90,000$개를 얻습니다. 여기서 총 $90,000$개의 누적 확률 값 중에서 top-k개를 뽑아 다음 time-step의 예측을 위한 디코더의 입력으로 사용하게 됩니다.

### Length Penalty

위의 탐색 알고리즘을 직접 짜서 수행시켜 보면 한가지 문제점이 발견됩니다. 현재 time-step 까지의 확률을 모두 곱(로그 확률의 경우에는 합)하기 때문에 문장이 길어질 수록 확률이 낮아진다는 점 입니다. 따라서 짧은 문장일수록 더 높은 점수를 획득하는 경향이 있습니다. 우리는 이러한 현상을 방지하기 위해서 예측한 문장의 길이에 따른 페널티를 주어 탐색이 조기 종료되는 것을 막습니다. 수식은 아래와 같습니다. 불행히도 우리는 2개의 추가적인 hyper-parameter가 필요합니다.

$$
\begin{gathered}
\log{\tilde{P}(\hat{Y}|X)}=\log{P(\hat{Y}|X)}\times\text{penalty} \\
\text{penalty}=\frac{(1+\text{length})^\alpha}{(1+\beta)^\alpha} \\
\text{where }\beta\text{ is hyper-parameter of minimum length}.
\end{gathered}
$$

## Code

### SingleBeamSearchSpace Class

![](../assets/nmt-beam-search-space.png)

#### Initialization

```python
from operator import itemgetter

import torch
import torch.nn as nn

import data_loader

LENGTH_PENALTY = 1.2
MIN_LENGTH = 5

class SingleBeamSearchSpace():

    def __init__(self, hidden, h_t_tilde = None, beam_size = 5, max_length = 255):
        self.beam_size = beam_size
        self.max_length = max_length

        super(SingleBeamSearchSpace, self).__init__()

        # To put data to same device.
        self.device = hidden[0].device
        # Inferred word index for each time-step. For now, initialized with initial time-step.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        # Index origin of current beam.
        self.prev_beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # 1 if it is done else 0
        self.masks = [torch.ByteTensor(beam_size).zero_().to(self.device)] 

        # We don't need to remember every time-step of hidden states: prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.
        # Future work: make this class to deal with any necessary information for other architecture, such as Transformer.

        # |hidden[0]| = (n_layers, 1, hidden_size)
        self.prev_hidden = torch.cat([hidden[0]] * beam_size, dim = 1)
        self.prev_cell = torch.cat([hidden[1]] * beam_size, dim = 1)
        # |prev_hidden| = (n_layers, beam_size, hidden_size)
        # |prev_cell| = (n_layers, beam_size, hidden_size)

        # |h_t_tilde| = (batch_size = 1, 1, hidden_size)
        self.prev_h_t_tilde = torch.cat([h_t_tilde] * beam_size, dim = 0) if h_t_tilde is not None else None
        # |prev_h_t_tilde| = (beam_size, 1, hidden_size)

        self.current_time_step = 0
        self.done_cnt = 0
        
    def get_length_penalty(self, length, alpha = LENGTH_PENALTY, min_length = MIN_LENGTH):
        # Calculate length-penalty, because shorter sentence usually have bigger probability.
        # Thus, we need to put penalty for shorter one.
        p = (1 + length) ** alpha / (1 + min_length) ** alpha

        return p

    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        hidden = (self.prev_hidden, self.prev_cell)
        h_t_tilde = self.prev_h_t_tilde

        # |y_hat| = (beam_size, 1)
        # |hidden| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size) or None
        return y_hat, hidden, h_t_tilde

    def collect_result(self, y_hat, hidden, h_t_tilde):
        # |y_hat| = (beam_size, 1, output_size)
        # |hidden| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size)
        output_size = y_hat.size(-1)

        self.current_time_step += 1

        # Calculate cumulative log-probability.
        # First, fill -inf value to last cumulative probability, if the beam is already finished.
        # Second, expand -inf filled cumulative probability to fit to 'y_hat'. (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size)
        # Third, add expanded cumulative probability to 'y_hat'
        cumulative_prob = y_hat + self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf')).view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # Now, we have new top log-probability and its index. We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.
        top_log_prob, top_indice = torch.topk(cumulative_prob.view(-1), self.beam_size, dim = -1)
        # |top_log_prob| = (beam_size)
        # |top_indice| = (beam_size)

        self.word_indice += [top_indice.fmod(output_size)] # Because we picked from whole batch, original word index should be calculated again.
        self.prev_beam_indice += [top_indice.div(output_size).long()] # Also, we can get an index of beam, which has top-k log-probability search result.
        
        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)] # Set finish mask if we got EOS.
        self.done_cnt += self.masks[-1].float().sum() # Calculate a number of finished beams.

        # Set hidden states for next time-step, using 'index_select' method.
        self.prev_hidden = torch.index_select(hidden[0], dim = 1, index = self.prev_beam_indice[-1]).contiguous()
        self.prev_cell = torch.index_select(hidden[1], dim = 1, index = self.prev_beam_indice[-1]).contiguous()
        self.prev_h_t_tilde = torch.index_select(h_t_tilde, dim = 0, index = self.prev_beam_indice[-1]).contiguous()

    def get_n_best(self, n = 1):
        sentences = []
        probs = []
        founds = []

        for t in range(len(self.word_indice)): # for each time-step,
            for b in range(self.beam_size): # for each beam,
                if self.masks[t][b] == 1: # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] / self.get_length_penalty(t)]
                    founds += [(t, b)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b]]
                    founds += [(t, b)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(zip(founds, probs), 
                                            key = itemgetter(1), 
                                            reverse = True
                                            )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.prev_beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
```

#### Initialization

```python
    def __init__(self, hidden, h_t_tilde = None, beam_size = 5, max_length = 255):
        self.beam_size = beam_size
        self.max_length = max_length

        super(SingleBeamSearchSpace, self).__init__()

        # To put data to same device.
        self.device = hidden[0].device
        # Inferred word index for each time-step. For now, initialized with initial time-step.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        # Index origin of current beam.
        self.prev_beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # 1 if it is done else 0
        self.masks = [torch.ByteTensor(beam_size).zero_().to(self.device)] 

        # We don't need to remember every time-step of hidden states: prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.
        # Future work: make this class to deal with any necessary information for other architecture, such as Transformer.

        # |hidden[0]| = (n_layers, 1, hidden_size)
        self.prev_hidden = torch.cat([hidden[0]] * beam_size, dim = 1)
        self.prev_cell = torch.cat([hidden[1]] * beam_size, dim = 1)
        # |prev_hidden| = (n_layers, beam_size, hidden_size)
        # |prev_cell| = (n_layers, beam_size, hidden_size)

        # |h_t_tilde| = (batch_size = 1, 1, hidden_size)
        self.prev_h_t_tilde = torch.cat([h_t_tilde] * beam_size, dim = 0) if h_t_tilde is not None else None
        # |prev_h_t_tilde| = (beam_size, 1, hidden_size)

        self.current_time_step = 0
        self.done_cnt = 0
```

#### Implement Length Penalty

```python        
    def get_length_penalty(self, length, alpha = LENGTH_PENALTY, min_length = MIN_LENGTH):
        # Calculate length-penalty, because shorter sentence usually have bigger probability.
        # Thus, we need to put penalty for shorter one.
        p = (1 + length) ** alpha / (1 + min_length) ** alpha

        return p
```

#### Mark if is done

```python
    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0
```

#### Generate Fabricated Mini-batch

```python
    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        hidden = (self.prev_hidden, self.prev_cell)
        h_t_tilde = self.prev_h_t_tilde

        # |y_hat| = (beam_size, 1)
        # |hidden| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size) or None
        return y_hat, hidden, h_t_tilde
```

#### Collect the Result and Pick Top-K

```python
    def collect_result(self, y_hat, hidden, h_t_tilde):
        # |y_hat| = (beam_size, 1, output_size)
        # |hidden| = (n_layers, beam_size, hidden_size)
        # |h_t_tilde| = (beam_size, 1, hidden_size)
        output_size = y_hat.size(-1)

        self.current_time_step += 1

        # Calculate cumulative log-probability.
        # First, fill -inf value to last cumulative probability, if the beam is already finished.
        # Second, expand -inf filled cumulative probability to fit to 'y_hat'. (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size)
        # Third, add expanded cumulative probability to 'y_hat'
        cumulative_prob = y_hat + self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf')).view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # Now, we have new top log-probability and its index. We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.
        top_log_prob, top_indice = torch.topk(cumulative_prob.view(-1), self.beam_size, dim = -1)
        # |top_log_prob| = (beam_size)
        # |top_indice| = (beam_size)

        self.word_indice += [top_indice.fmod(output_size)] # Because we picked from whole batch, original word index should be calculated again.
        self.prev_beam_indice += [top_indice.div(output_size).long()] # Also, we can get an index of beam, which has top-k log-probability search result.
        
        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)] # Set finish mask if we got EOS.
        self.done_cnt += self.masks[-1].float().sum() # Calculate a number of finished beams.

        # Set hidden states for next time-step, using 'index_select' method.
        self.prev_hidden = torch.index_select(hidden[0], dim = 1, index = self.prev_beam_indice[-1]).contiguous()
        self.prev_cell = torch.index_select(hidden[1], dim = 1, index = self.prev_beam_indice[-1]).contiguous()
        self.prev_h_t_tilde = torch.index_select(h_t_tilde, dim = 0, index = self.prev_beam_indice[-1]).contiguous()
```

#### Back-trace the History

```python
    def get_n_best(self, n = 1):
        sentences = []
        probs = []
        founds = []

        for t in range(len(self.word_indice)): # for each time-step,
            for b in range(self.beam_size): # for each beam,
                if self.masks[t][b] == 1: # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] / self.get_length_penalty(t)]
                    founds += [(t, b)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b]]
                    founds += [(t, b)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(zip(founds, probs), 
                                            key = itemgetter(1), 
                                            reverse = True
                                            )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.prev_beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
```

### Mini-batch Parallelized Beam-Search

![](../assets/nmt-mini-batch-parallelized-beam-search-overview.png)

```python
    def batch_beam_search(self, src, beam_size = 5, max_length = 255, n_best = 1):
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        h_0_tgt, c_0_tgt = h_0_tgt
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        h_0_tgt = (h_0_tgt, c_0_tgt)

        # initialize 'SingleBeamSearchSpace' as many as batch_size
        spaces = [SingleBeamSearchSpace((h_0_tgt[0][:, i, :].unsqueeze(1),
                                            h_0_tgt[1][:, i, :].unsqueeze(1)), 
                                            None, 
                                            beam_size, 
                                            max_length = max_length
                                            ) for i in range(batch_size)]
        done_cnt = [space.is_done() for space in spaces]

        length = 0
        # Run loop while sum of 'done_cnt' is smaller than batch_size, or length is still smaller than max_length.
        while sum(done_cnt) < batch_size and length <= max_length:
            # current_batch_size = sum(done_cnt) * beam_size

            # Initialize fabricated variables.
            # As far as batch-beam-search is running, 
            # temporary batch-size for fabricated mini-batch is 'beam_size'-times bigger than original batch_size.
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
            fab_h_src, fab_mask = [], []
            
            # Build fabricated mini-batch in non-parallel way.
            # This may cause a bottle-neck.
            for i, space in enumerate(spaces):
                if space.is_done() == 0: # Batchfy only if the inference for the sample is still not finished.
                    y_hat_, (hidden_, cell_), h_t_tilde_ = space.get_batch()

                    fab_input += [y_hat_]
                    fab_hidden += [hidden_]
                    fab_cell += [cell_]
                    if h_t_tilde_ is not None:
                        fab_h_t_tilde += [h_t_tilde_]
                    else:
                        fab_h_t_tilde = None

                    fab_h_src += [h_src[i, :, :]] * beam_size
                    fab_mask += [mask[i, :]] * beam_size

            # Now, concatenate list of tensors.
            fab_input = torch.cat(fab_input, dim = 0)
            fab_hidden = torch.cat(fab_hidden, dim = 1)
            fab_cell = torch.cat(fab_cell, dim = 1)
            if fab_h_t_tilde is not None:
                fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim = 0)
            fab_h_src = torch.stack(fab_h_src)
            fab_mask = torch.stack(fab_mask)
            # |fab_input| = (current_batch_size, 1)
            # |fab_hidden| = (n_layers, current_batch_size, hidden_size)
            # |fab_cell| = (n_layers, current_batch_size, hidden_size)
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            # |fab_h_src| = (current_batch_size, length, hidden_size)
            # |fab_mask| = (current_batch_size, length)

            emb_t = self.emb_dec(fab_input)
            # |emb_t| = (current_batch_size, 1, word_vec_dim)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t, fab_h_t_tilde, (fab_hidden, fab_cell))
            # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            # |context_vector| = (current_batch_size, 1, hidden_size)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output, context_vector], dim = -1)))
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            y_hat = self.generator(fab_h_t_tilde)
            # |y_hat| = (current_batch_size, 1, output_size)

            # separate the result for each sample.
            cnt = 0
            for space in spaces:
                if space.is_done() == 0:
                    # Decide a range of each sample.
                    from_index = cnt * beam_size
                    to_index = from_index + beam_size

                    # pick k-best results for each sample.
                    space.collect_result(y_hat[from_index:to_index],
                                                (fab_hidden[:, from_index:to_index, :], 
                                                    fab_cell[:, from_index:to_index, :]),
                                                fab_h_t_tilde[from_index:to_index]
                                                )
                    cnt += 1

            done_cnt = [space.is_done() for space in spaces]
            length += 1

        # pick n-best hypothesis.
        batch_sentences = []
        batch_probs = []

        # Collect the results.
        for i, space in enumerate(spaces):
            sentences, probs = space.get_n_best(n_best)

            batch_sentences += [sentences]
            batch_probs += [probs]

        return batch_sentences, batch_probs
```