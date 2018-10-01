# Implementation

아래는 텍스트 분류를 위한 학습 및 추론 프로그램 전체 소스코드 입니다. 최신 소스코드는 저자의 깃허브에서 다운로드 받을 수 있습니다.

- Github Repo: https://github.com/kh-kim/simple-ntc

## Usage

## Evaluation

## Code

### train.py

```python
import argparse

import torch
import torch.nn as nn

from data_loader import DataLoader

from simple_ntc.rnn import RNNClassifier
from simple_ntc.cnn import CNNClassifier
from simple_ntc.trainer import Trainer


def define_argparser():
    '''
    Define argument parser to handle parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--min_vocab_freq', type=int, default=2)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--early_stop', type=int, default=-1)

    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--word_vec_dim', type=int, default=128)
    p.add_argument('--hidden_size', type=int, default=256)

    p.add_argument('--rnn', action='store_true')
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--window_sizes', type=str, default='3,4,5')
    p.add_argument('--n_filters', type=str, default='100,100,100')

    config = p.parse_args()

    config.window_sizes = list(map(int, config.window_sizes.split(',')))
    config.n_filters = list(map(int, config.n_filters.split(',')))

    return config


def main(config):
    '''
    The main method of the program to train text classification.
    :param config: configuration from argument parser.
    '''
    dataset = DataLoader(train_fn=config.train,
                         valid_fn=config.valid,
                         batch_size=config.batch_size,
                         min_freq=config.min_vocab_freq,
                         max_vocab=config.max_vocab_size,
                         device=config.gpu_id
                         )

    vocab_size = len(dataset.text.vocab)
    n_classes = len(dataset.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(input_size=vocab_size,
                              word_vec_dim=config.word_vec_dim,
                              hidden_size=config.hidden_size,
                              n_classes=n_classes,
                              n_layers=config.n_layers,
                              dropout_p=config.dropout
                              )
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        # Train until converge
        rnn_trainer = Trainer(model, crit)
        rnn_trainer.train(dataset.train_iter,
                          dataset.valid_iter,
                          batch_size=config.batch_size,
                          n_epochs=config.n_epochs,
                          early_stop=config.early_stop,
                          verbose=config.verbose
                          )
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(input_size=vocab_size,
                              word_vec_dim=config.word_vec_dim,
                              n_classes=n_classes,
                              dropout_p=config.dropout,
                              window_sizes=config.window_sizes,
                              n_filters=config.n_filters
                              )
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        # Train until converge
        cnn_trainer = Trainer(model, crit)
        cnn_trainer.train(dataset.train_iter,
                          dataset.valid_iter,
                          batch_size=config.batch_size,
                          n_epochs=config.n_epochs,
                          early_stop=config.early_stop,
                          verbose=config.verbose
                          )

    torch.save({'rnn': rnn_trainer.best if config.rnn else None,
                'cnn': cnn_trainer.best if config.cnn else None,
                'config': config,
                'vocab': dataset.text.vocab,
                'classes': dataset.label.vocab
                }, config.model)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
```

### data_loader.py

```python
from torchtext import data


class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(self, train_fn, valid_fn, 
                 batch_size=64, 
                 device=-1, 
                 max_vocab=999999, 
                 min_freq=1,
                 use_eos=False, 
                 shuffle=True
                 ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param valid_fn: Valid-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super(DataLoader, self).__init__()

        # Define field of the input file.
        # The input file consists of two fields.
        self.label = data.Field(sequential=False,
                                use_vocab=True,
                                unk_token=None
                                )
        self.text = data.Field(use_vocab=True, 
                               batch_first=True, 
                               include_lengths=False, 
                               eos_token='<EOS>' if use_eos else None
                               )

        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # We would have two separate input file: train_fn, valid_fn
        # Files consist of two columns: label field and text field.
        train, valid = data.TabularDataset.splits(path='', 
                                                  train=train_fn, 
                                                  validation=valid_fn, 
                                                  format='tsv', 
                                                  fields=[('label', self.label),
                                                          ('text', self.text)
                                                          ]
                                                  )

        # Those loaded dataset would be feeded into each iterator:
        # train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.
        self.train_iter, self.valid_iter = data.BucketIterator.splits((train, valid), 
                                                                      batch_size=batch_size, 
                                                                      device='cuda:%d' % device if device >= 0 else 'cpu', 
                                                                      shuffle=shuffle,
                                                                      sort_key=lambda x: len(x.text),
                                                                      sort_within_batch=True
                                                                      )

        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
```

### trainer.py

```python
from tqdm import tqdm
import torch

import utils

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class Trainer():

    def __init__(self, model, crit):
        self.model = model
        self.crit = crit

        super().__init__()

        self.best = {}

    def get_best_model(self):
        self.model.load_state_dict(self.best['model'])

        return self.model

    def get_loss(self, y_hat, y, crit=None):
        crit = self.crit if crit is None else crit
        loss = crit(y_hat, y)

        return loss

    def train_epoch(self, 
                    train, 
                    optimizer, 
                    batch_size=64, 
                    verbose=VERBOSE_SILENT
                    ):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_param_norm, total_grad_norm = 0, 0, 0
        avg_loss, avg_param_norm, avg_grad_norm = 0, 0, 0
        sample_cnt = 0

        progress_bar = tqdm(train, 
                            desc='Training: ', 
                            unit='batch'
                            ) if verbose is VERBOSE_BATCH_WISE else train
        # Iterate whole train-set.
        for idx, mini_batch in enumerate(progress_bar):
            x, y = mini_batch.text, mini_batch.label
            # Don't forget make grad zero before another back-prop.
            optimizer.zero_grad()

            y_hat = self.model(x)

            loss = self.get_loss(y_hat, y)
            loss.backward()

            total_loss += loss
            total_param_norm += utils.get_parameter_norm(self.model.parameters())
            total_grad_norm += utils.get_grad_norm(self.model.parameters())

            # Caluclation to show status
            avg_loss = total_loss / (idx + 1)
            avg_param_norm = total_param_norm / (idx + 1)
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e' % (avg_param_norm,
                                                                                        avg_grad_norm,
                                                                                        avg_loss
                                                                                        ))

            optimizer.step()

            sample_cnt += mini_batch.text.size(0)
            if sample_cnt >= len(train.dataset.examples):
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, avg_param_norm, avg_grad_norm

    def train(self, 
              train, 
              valid, 
              batch_size=64,
              n_epochs=100, 
              early_stop=-1, 
              verbose=VERBOSE_SILENT
              ):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set, 
        early stopping will be executed if the requirement is satisfied.
        '''
        optimizer = torch.optim.Adam(self.model.parameters())

        lowest_loss = float('Inf')
        lowest_after = 0

        progress_bar = tqdm(range(n_epochs), 
                            desc='Training: ', 
                            unit='epoch'
                            ) if verbose is VERBOSE_EPOCH_WISE else range(n_epochs)
        for idx in progress_bar:  # Iterate from 1 to n_epochs
            if verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1, 
                                                             len(progress_bar), 
                                                             lowest_loss
                                                             ))
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train, 
                                                                             optimizer, 
                                                                             batch_size=batch_size, 
                                                                             verbose=verbose
                                                                             )
            _, avg_valid_loss = self.validate(valid, 
                                              verbose=verbose
                                              )

            # Print train status with different verbosity.
            if verbose is VERBOSE_EPOCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (float(avg_param_norm),
                                                                                                                                  float(avg_grad_norm),
                                                                                                                                  float(avg_train_loss),
                                                                                                                                  float(avg_valid_loss),
                                                                                                                                  float(lowest_loss)
                                                                                                                                  ))

            if avg_valid_loss < lowest_loss:
                # Update if there is an improvement.
                lowest_loss = avg_valid_loss
                lowest_after = 0

                self.best = {'model': self.model.state_dict(),
                             'optim': optimizer,
                             'epoch': idx,
                             'lowest_loss': lowest_loss
                             }
            else:
                lowest_after += 1

                if lowest_after >= early_stop and early_stop > 0:
                    break
        if verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()

    def validate(self, 
                 valid, 
                 crit=None, 
                 batch_size=256, 
                 verbose=VERBOSE_SILENT
                 ):
        '''
        Validate a model with given valid iterator.
        '''
        # We don't need to back-prop for these operations.
        with torch.no_grad():
            total_loss, total_correct, sample_cnt = 0, 0, 0
            progress_bar = tqdm(valid, 
                                desc='Validation: ', 
                                unit='batch'
                                ) if verbose is VERBOSE_BATCH_WISE else valid

            y_hats = []
            self.model.eval()
            # Iterate for whole valid-set.
            for idx, mini_batch in enumerate(progress_bar):
                x, y = mini_batch.text, mini_batch.label
                y_hat = self.model(x)
                # |y_hat| = (batch_size, n_classes)

                loss = self.get_loss(y_hat, y, crit)

                total_loss += loss
                sample_cnt += mini_batch.text.size(0)
                total_correct += float(y_hat.topk(1)[1].view(-1).eq(y).sum())

                avg_loss = total_loss / (idx + 1)
                y_hats += [y_hat]

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid_loss=%.4e accuarcy=%.4f' % (avg_loss, total_correct / sample_cnt))

                if sample_cnt >= len(valid.dataset.examples):
                    break
            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            y_hats = torch.cat(y_hats, dim=0)

            return y_hats, avg_loss
```

### rnn.py

```python
import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(self, 
                 input_size, 
                 word_vec_dim, 
                 hidden_size, 
                 n_classes,
                 n_layers=4, 
                 dropout_p=.3
                 ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_dim)
        self.rnn = nn.LSTM(input_size=word_vec_dim,
                           hidden_size=hidden_size,
                           num_layers=n_layers,
                           dropout=dropout_p,
                           batch_first=True,
                           bidirectional=True
                           )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_dim)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)

        return y
```

### cnn.py

```python
import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 n_classes,
                 dropout_p=.5,
                 window_sizes=[3, 4, 5],
                 n_filters=[100, 100, 100]
                 ):
        self.input_size = input_size  # vocabulary size
        self.word_vec_dim = word_vec_dim
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        # window_size means that how many words a pattern covers.
        self.window_sizes = window_sizes
        # n_filters means that how many patterns to cover.
        self.n_filters = n_filters

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_dim)
        # Since number of convolution layers would be vary depend on len(window_sizes),
        # we use 'setattr' and 'getattr' methods to add layers to nn.Module object.
        for window_size, n_filter in zip(window_sizes, n_filters):
            cnn = nn.Conv2d(in_channels=1,
                            out_channels=n_filter,
                            kernel_size=(window_size, word_vec_dim)
                            )
            setattr(self, 'cnn-%d-%d' % (window_size, n_filter), cnn)
        # Because below layers are just operations, 
        # (it does not have learnable parameters)
        # we just declare once.
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        # An input of generator layer is max values from each filter.
        self.generator = nn.Linear(sum(n_filters), n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_dim)
        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            # Because some input does not long enough for maximum length of window size,
            # we add zero tensor for padding.
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_dim).zero_()
            # |pad| = (batch_size, min_length - length, word_vec_dim)
            x = torch.cat([x, pad], dim=1)
            # |x| = (batch_size, min_length, word_vec_dim)

        # In ordinary case of vision task, you may have 3 channels on tensor,
        # but in this case, you would have just 1 channel,
        # which is added by 'unsqueeze' method in below:
        x = x.unsqueeze(1)
        # |x| = (batch_size, 1, length, word_vec_dim)

        cnn_outs = []
        for window_size, n_filter in zip(self.window_sizes, self.n_filters):
            cnn = getattr(self, 'cnn-%d-%d' % (window_size, n_filter))
            cnn_out = self.dropout(self.relu(cnn(x)))
            # |x| = (batch_size, n_filter, length - window_size + 1, 1)

            # In case of max pooling, we does not know the pooling size,
            # because it depends on the length of the sentence.
            # Therefore, we use instant function using 'nn.functional' package.
            # This is the beauty of PyTorch. :)
            cnn_out = nn.functional.max_pool1d(input=cnn_out.squeeze(-1),
                                               kernel_size=cnn_out.size(-2)
                                               ).squeeze(-1)
            # |cnn_out| = (batch_size, n_filter)
            cnn_outs += [cnn_out]
        # Merge output tensors from each convolution layer.
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        # |cnn_outs| = (batch_size, sum(n_filters))
        y = self.activation(self.generator(cnn_outs))
        # |y| = (batch_size, n_classes)

        return y
```

### utils.py

```python
def get_grad_norm(parameters, norm_type=2):
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


def get_parameter_norm(parameters, norm_type=2):
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

### classify.py

```python
import sys
import argparse

import torch
import torch.nn as nn
from torchtext import data

from simple_ntc.rnn import RNNClassifier
from simple_ntc.cnn import CNNClassifier

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_text():
    # This method gets sentences from standard input and tokenize those.
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

    return lines


def define_field():
    return data.Field(use_vocab=True, 
                      batch_first=True, 
                      include_lengths=False
                      ), data.Field(sequential=False, use_vocab=True, unk_token=None)


def main(config):
    saved_data = torch.load(config.model)

    train_config = saved_data['config']

    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text()

    with torch.no_grad():
        # Converts string to list of index.
        x = text_field.numericalize(text_field.pad(lines),
                                          device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
                                          )

        ensemble = []
        if rnn_best is not None:
            model = RNNClassifier(input_size=vocab_size,
                                  word_vec_dim=train_config.word_vec_dim,
                                  hidden_size=train_config.hidden_size,
                                  n_classes=n_classes,
                                  n_layers=train_config.n_layers,
                                  dropout_p=train_config.dropout
                                  )
            model.load_state_dict(rnn_best['model'])
            ensemble += [model]
        if cnn_best is not None:
            model = CNNClassifier(input_size=vocab_size,
                                  word_vec_dim=train_config.word_vec_dim,
                                  n_classes=n_classes,
                                  dropout_p=train_config.dropout,
                                  window_sizes=train_config.window_sizes,
                                  n_filters=train_config.n_filters
                                  )
            model.load_state_dict(cnn_best['model'])
            ensemble += [model]

        y_hats = []
        for model in ensemble:
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            model.eval()

            y_hat = []
            for idx in range(0, len(lines), config.batch_size):
                y_hat += [model(x[idx:idx + config.batch_size])]
            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)

            y_hats += [y_hat]
        y_hats = torch.stack(y_hats).exp()
        # |y_hats| = (len(ensemble), len(lines), n_classes)
        y_hats = y_hats.sum(dim=0) / len(ensemble)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)]), ' '.join(lines[i])))

if __name__ == '__main__':
    config = define_argparser()
    main(config)
```