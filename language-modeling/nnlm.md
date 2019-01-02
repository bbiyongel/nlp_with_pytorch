# 뉴럴 네트워크 언어 모델링 (nNeural nNetwork lLanguage mModeling)

## 희소성을 해결하기 위하여Against to Sparseness

앞서 설명한 것과 같이 기존의 n-gram 기반의 언어모델은 간편하지만 훈련 데이터에서 보지 못한 단어의 조합에 대해서 상당히 취약한 부분이 있었습니다. 그것의 근본적인 원인은 n-gram 기반의 언어모델은 단어간의 유사도를 알 지 못하기 때문입니다. 예를 들어 우리에게 훈련 코퍼스corpus로 아래와 같은 문장이 주어졌다고 했을 때,

* 고양이는 좋은 반려동물 입니다.

사람은 각 단어간의 유사도를 알기 때문에 다음 중 어떤 확률이 더 큰지 알 수 있습니다.

* $P(\text{반려동물}|\text{강아지는}, \text{좋은})$
* $P(\text{반려동물}|\text{자동차는}, \text{좋은})$

 하지만, 컴퓨터는 훈련 코퍼스corpus에 해당 n-gram이 존재하지 않으면, 출현빈도를 계산count를 할 수 없기 때문에 확률을 구할 수 없고, 따라서 확률 간 비교를 할 수도 없습니다.

* $P(\text{반려동물}|\text{강아지는}, \text{좋은})$
* $P(\text{반려동물}|\text{자동차는}, \text{좋은})$

비록 강아지가 개의 새끼이고 포유류에 속하는 가축에 해당한다는 깊고 해박한 지식이 없을지라도, 강아지와 고양이 사이의 유사도가 자동차와 고양이 사이의 유사도보다 높은 것을 알기 때문에 자동차 보다는 강아지에 대한 반려동물의 확률이 더 높음을 유추할 수 있습니다. 하지만 n-gram 방식의 언어모델은 단어간의 유사도를 구할 수 없기 때문에, 이와 같이 훈련 코퍼스corpus에서 보지 못한 단어(unseen word sequence)의 조합(n-gram)에 대해서 효과적으로 대처할 수 없습니다.

하지만 뉴럴 네트워크 언어모델(NNNeural Network LM)은 단어 임베딩을 사용하여 단어를 차원축소word embedding을 사용하여 단어를 벡터화(vectorize) 함으로써, 강아지와 고양이를 비슷한 dense 벡터vector로 학습하고, 자동차와 고양이 보다 훨씬 높은 유사도를 가지게 합니다. 따라서 NNLM이 훈련 코퍼스corpus에서 보지 못한 단어의 조합을 보더라도, 비슷한 훈련 데이터로부터 배운 것과 유사하게 대처할 수 있습니다. 즉, 희소성 해소를 통해 더 좋은 일반화(generalization) 성능을 얻어낼 수 있습니다.

NN

Neural Network LM은 다양한많은 형태를 가질 수 있지만 우리는 가장 효율적이고 흔한 형태인 Recurrent Neural Network(RNN)의 일종인 Long Short Term Memory(LSTM)을 활용한 방식에 대해서 짚고 넘어가도록 하겠습니다.

## Recurrent Neural Network 언어모델LM

![Recurrent Neural 언어모델 구조Language Model 아키텍처](../assets/rnn_lm_architecture.png)

Recurrent Neural Network 언어모델Lauguage Model (RNNLM)은 위와 같은 구조를 지니고 있습니다. 기존의 언어모델은 각각의 단어를 descrete한 데이터로 취급존재로써 처리하였기 때문에, 단어 시퀀스문장(word sequence)의 길이가 길어지면 희소성(sparseness)문제가 발생하여 어려움을 겪운 부분이 있었습니다. 따라서 마코프 가정을 통해, $n-1$ 이전까지의 단어만 (주로 $n=3$) 조건부로 사용하여잡아 확률을 근사(approximation) 하였습니다. 하지만, RNN LM은 단어 임베딩을 통해 dense 벡터로 만듦를 embedding하여 벡터화(vectorize)함으로써, 희소성 문제를 해소하였기 때문에, 문장의 첫 단어부터 해당 단어 직전의 단어까지 모두 조건부에 넣어 확률을 근사 할 수 있습니다.

$$
P(w_1,w_2,\cdots,w_k) = \prod_{i=1}^{k}{P(w_i|w_{<i})}
$$

로그를 취하여 합으로 표현하표현 해보면 아래와 같습니다.

$$
\log{P(w_1, w_2, \cdots, w_k)} = \sum_{i = 1}^{k}{\log{P(w_i|w_{<i})}}
$$

## 구현Implementation

이제 RNN을 활용한 언어모델을 구현 해 보도록 하겠습니다. 파이토치PyTorch로 구현하기에 앞서, 이를 수식화 해보면 아래와 같습니다. <comment> language_model.py가 이를 구현 한 코드 입니다. </comment>

$$
\begin{gathered}
x_{1:n}X=\{x_0,x_1,\cdots,x_n,x_{n+1}\} \\
\text{where }x_0=\text{BOS}\text{ and }x_{n+1}=\text{EOS}. \\ \\
\hat{x}_{i+1}=\text{softmax}(\text{linear}_{\text{hidden\_size}hidden \rightarrow |V|}(\text{RNN}(\text{emb}(x_i)))) \\
\hat{x}_{1:nX}[1:]=\text{softmax}(\text{linear}_{\text{hidden\_size}hidden \rightarrow |V|}(\text{RNN}(\text{emb}(x_{1:n}X[:-1])))), \\
\\
\text{linear}_{d_1\rightarrow d_2}(x)=Wx+b \text{ where }W\in\mathbb{R}^{d_1\times d_2}\text{ and }b\in\mathbb{R}^{d_2},\\
\text{and hidden\_size is dimension of hidden state and }|V|\text{ is size of vocabulary}.
\end{gathered}
$$

이때 입력 문장의 시작과 끝에는 $x_0$과 $x_{n+1}$이 추가 되어 BOS와 EOS를 나타냅니다. 따라서 실제 문장을 나타내는 시퀀스의 길이는 2만큼 더 늘어납니다.

수식을 과정 별로where }|V|\text{ is size of vocabulary}.
\end{gathered}
$$

위의 수식을 따라가 보면, 먼저 문장 $x_{1:n}[:-1]X$를 입력으로 받아 각 time-step 별 토큰 ($x_i$)로 임베딩 레이어 EmbEmb(embedding layer)에 넣어 정해진 차원(dimension)의 단어 임베딩 벡터를 얻습니다. 여기서 주의할 점은 EOS를 떼고 임베딩 레이어에 입력으로 주어진다는 것 입니다.

$$
\begin{gathered}
x_{1:n}[:-1]=\{x_0,x_1,\cdots,x_n\} \\
x_{\text{emb}}=\text{emb}(x_{1:n}[:-1]) \\
\\
\text{where }|x_{1:n}[:-1]|=(\text{batch\_size},n+1) \\
\text{ and }|x_\text{emb}|=(\text{batch\_size},n+1,\text{word\_vec\_dim}).
\end{gathered}
$$

RNN은 해당 단어 임베딩 벡터를 입력으로 받아, RNN의 히든 스테이트의 크기인embedding vector를 얻습니다. RNN은 해당 embedding vector를 입력으로 받아, hidden_ size의 벡터를vector 형태로 반환 합니다. 이 과정은 파이토치를 통해 문장의 모든 time-step을 한번에 병렬로 계산할 수 있습니다.

$$
\begin{gathered}
h_{0:n}=\text{RNN}(x_{\text{emb}}) \\
\text{where }|h_{0:n}|=(\text{batch\_size},n+1,\text{hidden\_size})
\end{gathered}
$$

이 벡터를 리니어 레이어(linear layer)와 softmax 함수를 통해 각 단어에 대한 확률 분포인 $\hat{x}_{i+1}$을 구합니다.

$$
\begin{gathered}
\hat{x}_{1:n}[1:]=\text{softmax}(\text{linear}_{\text{hidden\_size} \rightarrow |V|}(h_{0:n})) \\
\\
\text{where }|\hat{x}_{1:n}[1:]|=(\text{batch\_size},n+1,|V|) \\
\text{and }x_{1:n}[1:]=\{x_1,x_2,\cdots,x_{n+1}\}
\end{gathered}
$$RNN 출력 vector를 linear layer를 통해 어휘(vocabulary)수 dimension의 vector로 변환 한 후, softmax를 취하여 $\hat{x}_{i+1}$을 구합니다.

여기서 우리는 LSTM을 사용하여 RNN을 대체 할 것이고, LSTM은 여러 층(layer)로 구성되어 있으며, 각 층 사이에는 드랍아웃(dropout)이 들어갈 수가 있습니다. 우리는 테스트 데이터셋에 대해서 perplexity를 최소화 하는 것이 목표이기 때문에, 이전 섹션에서 perplexity와 엔트로피(entropy)와의 관계를 설명하였듯이, 크로스 엔트로피 손실 함수를 사용하여 최적화를 수행 합니다. <comment> one-hot 벡터의 크로스 엔트로피 연산은 이전 기초 수학 챕터를 참고 바랍니다. </comment> 이때 주의할 점은 입력과 반대로 BOS를 제거한 정답 $x_{1:n}[1:]$와 비교한다는 것 입니다.

$$
\begin{gathered}
\mathcal{L}(\hat{x}_{1:n}[1:], x_{1:n}[1:])=-\frac{1}{m}\sum_{i=1}^m{\sum_{j=1}^n{x_j^i\log{\hat{x}_j^i}}} \\
\text{where }x_j^i\text{ is one-hot vector}.
\end{gathered}
$$

## 파이토치 예제 코드

아래의 파이토치이 결과($\hat{X}$)를 이전 섹션에서 perplexity와 엔트로피(entropy)와의 관계를 설명하였듯이, cross entropy loss를 사용하여 모델($\theta$) 최적화를 수행 합니다.

## Code

아래의 PyTorch 코드는 저자의 깃허브github에서 다운로드 할 수 있습니다. (업데이트 여부에 따라 코드가 약간 달라질 수 있습니다.)

- github repo url: https://github.com/kh-kim/OpenNLMTK

### language_model.py

```python
import torch
import torch.nn as nn

import data_loader


class LanguageModel(nn.Module):

    def __init__(self, 
                 vocab_size,
                 word_vec_dim=512,
                 hidden_size=512,
                 n_layers=4,
                 dropout_p=.2,
                 max_length=255
                 ):
        self.vocab_size = vocab_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        super(LanguageModel, self).__init__()

        self.emb = nn.Embedding(vocab_size, 
                                word_vec_dim,
                                padding_idx=data_loader.PAD
                                )
        self.rnn = nn.LSTM(word_vec_dim,
                           hidden_size,
                           n_layers,
                           batch_first=True,
                           dropout=dropout_p
                           )
        self.out = nn.Linear(hidden_size, vocab_size, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x) 
        # |x| = (batch_size, length, word_vec_dim)
        x, (h, c) = self.rnn(x) 
        # |x| = (batch_size, length, hidden_size)
        x = self.out(x) 
        # |x| = (batch_size, length, vocab_size)
        y_hat = self.log_softmax(x)

        return y_hat

    def search(self, batch_size=64, max_length=255):
        x = torch.LongTensor(batch_size, 1).to(next(self.parameters()).device).zero_() + data_loader.BOS
        # |x| = (batch_size, 1)
        is_undone = x.new_ones(batch_size, 1).float()

        y_hats, indice = [], []
        h, c = None, None
        while is_undone.sum() > 0 and len(indice) < max_length:
            x = self.emb(x)
            # |emb_t| = (batch_size, 1, word_vec_dim)

            x, (h, c) = self.rnn(x, (h, c)) if h is not None and c is not None else self.rnn(x)
            # |x| = (batch_size, 1, hidden_size)
            y_hat = self.log_softmax(x)
            # |y_hat| = (batch_size, 1, output_size)
            y_hats += [y_hat]

            # y = torch.topk(y_hat, 1, dim = -1)[1].squeeze(-1)
            y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
            y = y.masked_fill_((1. - is_undone).byte(), data_loader.PAD)
            is_undone = is_undone * torch.ne(y, data_loader.EOS).float()            
            # |y| = (batch_size, 1)
            # |is_undone| = (batch_size, 1)
            indice += [y]

            x = y

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice
```

## 결론# data_loader.py

```python
from torchtext import data, datasets

PAD, BOS, EOS = 1, 2, 3


class DataLoader():

    def __init__(self, 
                 train_fn,
                 valid_fn, 
                 batch_size=64, 
                 device='cpu', 
                 max_vocab=99999999, 
                 max_length=255, 
                 fix_length=None, 
                 use_bos=True, 
                 use_eos=True, 
                 shuffle=True
                 ):
        
        super(DataLoader, self).__init__()

        self.text = data.Field(sequential=True, 
                               use_vocab=True, 
                               batch_first=True, 
                               include_lengths=True, 
                               fix_length=fix_length, 
                               init_token='<BOS>' if use_bos else None, 
                               eos_token='<EOS>' if use_eos else None
                               )

        train = LanguageModelDataset(path=train_fn, 
                                     fields=[('text', self.text)], 
                                     max_length=max_length
                                     )
        valid = LanguageModelDataset(path=valid_fn, 
                                     fields=[('text', self.text)], 
                                     max_length=max_length
                                     )

        self.train_iter = data.BucketIterator(train, 
                                              batch_size=batch_size, 
                                              device='cuda:%d' % device if device >= 0 else 'cpu', 
                                              shuffle=shuffle, 
                                              sort_key=lambda x: -len(x.text), 
                                              sort_within_batch=True
                                              )
        self.valid_iter = data.BucketIterator(valid, 
                                              batch_size=batch_size, 
                                              device='cuda:%d' % device if device >= 0 else 'cpu', 
                                              shuffle=False, 
                                              sort_key=lambda x: -len(x.text), 
                                              sort_within_batch=True
                                              )

        self.text.build_vocab(train, max_size=max_vocab)


class LanguageModelDataset(data.Dataset):

    def __init__(self, path, fields, max_length=None, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('text', fields[0])]

        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if max_length and max_length < len(line.split()):
                    continue
                if line != '':
                    examples.append(data.Example.fromlist(
                        [line], fields))

        super(LanguageModelDataset, self).__init__(examples, fields, **kwargs)
```

### trainer.py

```python
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils


def get_loss(y, y_hat, criterion, do_backward=True):
    batch_size = y.size(0)

    loss = criterion(y_hat.contiguous().view(-1, y_hat.size(-1)), 
                     y.contiguous().view(-1)
                     )
    if do_backward:
        # since size_average parameter is off, we need to devide it by batch size before back-prop.
        loss.div(batch_size).backward()

    return loss


def train_epoch(model, criterion, train_iter, valid_iter, config):
    current_lr = config.lr

    lowest_valid_loss = np.inf
    no_improve_cnt = 0

    for epoch in range(1, config.n_epochs + 1):
        # optimizer = optim.Adam(model.parameters(), lr = current_lr)
        optimizer = optim.SGD(model.parameters(),
                              lr=current_lr
                              )
        print("current learning rate: %f" % current_lr)
        print(optimizer)

        sample_cnt = 0
        total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
        start_time = time.time()
        train_loss = np.inf

        for batch_index, batch in enumerate(train_iter):
            optimizer.zero_grad()

            current_batch_word_cnt = torch.sum(batch.text[1])
            # Most important lines in this method.
            # Since model takes BOS + sentence as an input and sentence + EOS as an output,
            # x(input) excludes last index, and y(index) excludes first index.
            x = batch.text[0][:, :-1]
            y = batch.text[0][:, 1:]
            # feed-forward
            y_hat = model(x)

            # calcuate loss and gradients with back-propagation
            loss = get_loss(y, y_hat, criterion)
            
            # simple math to show stats
            total_loss += float(loss)
            total_word_count += int(current_batch_word_cnt)
            total_parameter_norm += float(utils.get_parameter_norm(model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(model.parameters()))

            if (batch_index + 1) % config.print_every == 0:
                avg_loss = total_loss / total_word_count
                avg_parameter_norm = total_parameter_norm / config.print_every
                avg_grad_norm = total_grad_norm / config.print_every
                elapsed_time = time.time() - start_time

                print("epoch: %d batch: %d/%d\t|param|: %.2f\t|g_param|: %.2f\tloss: %.4f\tPPL: %.2f\t%5d words/s %3d secs" % (epoch, 
                                                                                                                               batch_index + 1, 
                                                                                                                               int((len(train_iter.dataset.examples) // config.batch_size)  * config.iter_ratio_in_epoch), 
                                                                                                                               avg_parameter_norm, 
                                                                                                                               avg_grad_norm, 
                                                                                                                               avg_loss,
                                                                                                                               np.exp(avg_loss),
                                                                                                                               total_word_count // elapsed_time,
                                                                                                                               elapsed_time
                                                                                                                               ))

                total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
                start_time = time.time()

                train_loss = avg_loss

            # Another important line in this method.
            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(model.parameters(), 
                                        config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += batch.text[0].size(0)
            if sample_cnt >= len(train_iter.dataset.examples) * config.iter_ratio_in_epoch:
                break

        sample_cnt = 0
        total_loss, total_word_count = 0, 0

        model.eval()
        for batch_index, batch in enumerate(valid_iter):
            current_batch_word_cnt = torch.sum(batch.text[1])
            x = batch.text[0][:, :-1]
            y = batch.text[0][:, 1:]
            y_hat = model(x)

            loss = get_loss(y, y_hat, criterion, do_backward=False)

            total_loss += float(loss)
            total_word_count += int(current_batch_word_cnt)

            sample_cnt += batch.text[0].size(0)
            if sample_cnt >= len(valid_iter.dataset.examples):
                break

        avg_loss = total_loss / total_word_count
        print("valid loss: %.4f\tPPL: %.2f" % (avg_loss, np.exp(avg_loss)))

        if lowest_valid_loss > avg_loss:
            lowest_valid_loss = avg_loss
            no_improve_cnt = 0
        else:
            # decrease learing rate if there is no improvement.
            current_lr = max(config.min_lr, current_lr * config.lr_decay_rate)
            no_improve_cnt += 1

        model.train()

        model_fn = config.model.split(".")
        model_fn = model_fn[:-1] + ["%02d" % epoch, 
                                    "%.2f-%.2f" % (train_loss, np.exp(train_loss)), 
                                    "%.2f-%.2f" % (avg_loss, np.exp(avg_loss))
                                    ] + [model_fn[-1]]

        # PyTorch provides efficient method for save and load model, which uses python pickle.
        torch.save({"model": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "current_lr": current_lr
                    }, ".".join(model_fn))

        if config.early_stop > 0 and no_improve_cnt >= config.early_stop:
            break
```

### utils.py

```python
def get_grad_norm(parameters, norm_type = 2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def get_parameter_norm(parameters, norm_type = 2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
```

### train.py

```python
import argparse

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from language_model import LanguageModel as LM
import trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required=True)
    p.add_argument('-train', required=True)
    p.add_argument('-valid', required=True)
    p.add_argument('-gpu_id', type=int, default=-1)

    p.add_argument('-batch_size', type=int, default=64)
    p.add_argument('-n_epochs', type=int, default=20)
    p.add_argument('-print_every', type=int, default=50)
    p.add_argument('-early_stop', type=int, default=3)
    p.add_argument('-iter_ratio_in_epoch', type=float, default=1.)
    p.add_argument('-lr_decay_rate', type=float, default=.5)
    
    p.add_argument('-dropout', type=float, default=.3)
    p.add_argument('-word_vec_dim', type=int, default=256)
    p.add_argument('-hidden_size', type=int, default=256)
    p.add_argument('-max_length', type=int, default=80)

    p.add_argument('-n_layers', type=int, default=4)
    p.add_argument('-max_grad_norm', type=float, default=5.)
    p.add_argument('-lr', type=float, default=1.)
    p.add_argument('-min_lr', type=float, default=.000001)

    p.add_argument('-gen', type=int, default=32)
    
    config = p.parse_args()

    return config


def to_text(indice, vocab):
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines


if __name__ == '__main__':
    config = define_argparser()

    loader = DataLoader(config.train, 
                        config.valid, 
                        batch_size=config.batch_size, 
                        device=config.gpu_id,
                        max_length=config.max_length
                        )
    model = LM(len(loader.text.vocab), 
               word_vec_dim=config.word_vec_dim, 
               hidden_size=config.hidden_size, 
               n_layers=config.n_layers, 
               dropout_p=config.dropout, 
               max_length=config.max_length
               )

    # Let criterion cannot count PAD as right prediction, because PAD is easy to predict.
    loss_weight = torch.ones(len(loader.text.vocab))
    loss_weight[data_loader.PAD] = 0
    criterion = nn.NLLLoss(weight=loss_weight, size_average=False)

    print(model)
    print(criterion)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        criterion.cuda(config.gpu_id)

    if config.n_epochs > 0:
        trainer.train_epoch(model, 
                            criterion, 
                            loader.train_iter, 
                            loader.valid_iter, 
                            config
                            )

    if config.gen > 0:
        total_gen = 0
        while total_gen < config.gen:
            current_gen = min(config.batch_size, config.gen - total_gen)
            _, indice = model.search(batch_size=current_gen)
            total_gen += current_gen

            lines = to_text(indice, loader.text.vocab)
            print('\n'.join(lines))
```

## Conclusion

NNLM은 word embedding vector를 사용하여 희소성(sparseness)을 해결하여 큰 효과를 보았습니다. 따라서 훈련 데이터셋에서 보지 못한 단어(unseen word sequence)의 조합에 대해서도 훌륭한 대처가 가능합니다. 하지만 그만큼 연산량에 있어서 n-gram에 비해서 매우 많은 대가를 치루어야 합니다. 단순히 table look-up 수준의 연산량을 필요로 하는 n-gram방식에 비해서 NNLM은 다수의 matrix 연산등이 포함된 feed forward 연산을 수행해야 하기 때문입니다. 그럼에도 불구하고 GPU의 사용과 점점 빨라지는 하드웨어 사이에서 NNLM의 중요성은 커지고 있고, 실제로도 많은 분야에 적용되어 훌륭한 성과를 거두고 있습니다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MTIzMjUwODVdfQ==
-->