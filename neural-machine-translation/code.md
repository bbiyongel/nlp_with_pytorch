# Full Source Code for Neural Machine Translation via RNN Sequence-to-Sequence

github repo url: https://github.com/kh-kim/simple-nmt

## train.py

```python
import argparse, sys

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
import simple_nmt.trainer as trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True, help = 'Model file name to save. Additional information would be annotated to the file name.')
    p.add_argument('-train', required = True, help = 'Training set file name except the extention. (ex: train.en --> train)')
    p.add_argument('-valid', required = True, help = 'Validation set file name except the extention. (ex: valid.en --> valid)')
    p.add_argument('-lang', required = True, help = 'Set of extention represents language pair. (ex: en + ko --> enko)')
    p.add_argument('-gpu_id', type = int, default = -1, help = 'GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default = -1')

    p.add_argument('-batch_size', type = int, default = 32, help = 'Mini batch size for gradient descent. Default = 32')
    p.add_argument('-n_epochs', type = int, default = 18, help = 'Number of epochs to train. Default = 18')
    p.add_argument('-print_every', type = int, default = 1000, help = 'Number of gradient descent steps to skip printing the training status. Default = 1000')
    p.add_argument('-early_stop', type = int, default = -1, help = 'The training will be stopped if there is no improvement this number of epochs. Default = -1')

    p.add_argument('-max_length', type = int, default = 80, help = 'Maximum length of the training sequence. Default = 80')
    p.add_argument('-dropout', type = float, default = .2, help = 'Dropout rate. Default = 0.2')
    p.add_argument('-word_vec_dim', type = int, default = 512, help = 'Word embedding vector dimension. Default = 512')
    p.add_argument('-hidden_size', type = int, default = 768, help = 'Hidden size of LSTM. Default = 768')
    p.add_argument('-n_layers', type = int, default = 4, help = 'Number of layers in LSTM. Default = 4')
    
    p.add_argument('-max_grad_norm', type = float, default = 5., help = 'Threshold for gradient clipping. Default = 5.0')
    p.add_argument('-adam', action = 'store_true', help = 'Use Adam instead of using SGD.')
    p.add_argument('-lr', type = float, default = 1., help = 'Initial learning rate. Default = 1.0')
    p.add_argument('-min_lr', type = float, default = .000001, help = 'Minimum learning rate. Default = .000001')
    p.add_argument('-lr_decay_start_at', type = int, default = 10, help = 'Start learning rate decay from this epoch.')
    p.add_argument('-lr_slow_decay', action = 'store_true', help = 'Decay learning rate only if there is no improvement on last epoch.')
    p.add_argument('-lr_decay_rate', type = float, default = .5, help = 'Learning rate decay rate. Default = 0.5')

    config = p.parse_args()

    return config

def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for key in vars(prev_config).keys():
            if '-%s' % key not in sys.argv or key == 'model':
                if vars(config).get(key) is not None:
                    vars(config)[key] = vars(prev_config)[key]
                else:
                    # Missing argument
                    print('WARNING!!! Argument "-%s" is not found in current argument parser.\tSaved value:' % key, vars(prev_config)[key])
            else:
                # Argument value is change from saved model.
                print('WARNING!!! Argument "-%s" is not loaded from saved model.\tCurrent value:' % key, vars(config)[key])

    return config


if __name__ == "__main__":
    config = define_argparser()

    import os.path
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.model):
        saved_data = torch.load(config.model)
    
        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)
        config.lr = saved_data['current_lr']
    else:
        saved_data = None
    
    # Load training and validation data set.
    loader = DataLoader(config.train, 
                        config.valid, 
                        (config.lang[:2], config.lang[-2:]), 
                        batch_size = config.batch_size, 
                        device = config.gpu_id, 
                        max_length = config.max_length
                        )

    input_size = len(loader.src.vocab) # Encoder's embedding layer input size
    output_size = len(loader.tgt.vocab) # Decoder's embedding layer input size and Generator's softmax layer output size
    # Declare the model
    model = Seq2Seq(input_size,
                    config.word_vec_dim, # Word embedding vector size
                    config.hidden_size, # LSTM's hidden vector size
                    output_size, 
                    n_layers = config.n_layers, # number of layers in LSTM
                    dropout_p = config.dropout # dropout-rate in LSTM
                    )

    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones(output_size)
    loss_weight[data_loader.PAD] = 0.
    # Instead of using Cross-Entropy loss, we can use Negative Log-Likelihood(NLL) loss with log-probability.
    criterion = nn.NLLLoss(weight = loss_weight, size_average = False) 

    print(model)
    print(criterion)

    # Pass models to GPU device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        criterion.cuda(config.gpu_id)

    # If we have loaded model weight parameters, use that weights for declared model.
    if saved_data is not None:
        model.load_state_dict(saved_data['model'])

    # Start training. This function maybe equivalant to 'fit' function in Keras.
    trainer.train_epoch(model, 
                        criterion, 
                        loader.train_iter, 
                        loader.valid_iter, 
                        config,
                        start_epoch = saved_data['epoch'] if saved_data is not None else 1,
                        others_to_save = {'src_vocab': loader.src.vocab, 'tgt_vocab': loader.tgt.vocab} # We can put any object here to save with model.
                        )
```

## translate.py

```python
import argparse, sys
from operator import itemgetter

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
import simple_nmt.trainer as trainer

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True, help = 'Model file name to use')
    p.add_argument('-gpu_id', type = int, default = -1, help = 'GPU ID to use. -1 for CPU. Default = -1')

    p.add_argument('-batch_size', type = int, default = 128, help = 'Mini batch size for parallel inference. Default = 128')
    p.add_argument('-max_length', type = int, default = 255, help = 'Maximum sequence length for inference. Default = 255')
    p.add_argument('-n_best', type = int, default = 1, help = 'Number of best inference result per sample. Default = 1')
    p.add_argument('-beam_size', type = int, default = 5, help = 'Beam size for beam search. Default = 5')
    
    config = p.parse_args()

    return config

def read_text():
    # This method gets sentences from standard input and tokenize those.
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

    return lines

def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                #line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines

if __name__ == '__main__':
    config = define_argparser()

    # Load saved model.
    saved_data = torch.load(config.model)
    
    # Load configuration setting in training.
    train_config = saved_data['config']
    # Load vocabularies from the model.
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']

    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)
    input_size = len(loader.src.vocab)
    output_size = len(loader.tgt.vocab)

    # Declare sequence-to-sequence model.
    model = Seq2Seq(input_size,
                    train_config.word_vec_dim,
                    train_config.hidden_size,
                    output_size,
                    n_layers = train_config.n_layers,
                    dropout_p = train_config.dropout
                    )
    model.load_state_dict(saved_data['model']) # Load weight parameters from the trained model.
    model.eval() # We need to turn-on the evaluation mode, which turns off all drop-outs.

    # We don't need to draw a computation graph, because we will have only inferences.
    torch.set_grad_enabled(False) 

    # Put models to device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    # Get sentences from standard input.
    lines = read_text()
    
    with torch.no_grad(): # Also, declare again to prevent to get gradients.
        while len(lines) > 0:
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            sorted_lines = lines[:config.batch_size]
            lines = lines[config.batch_size:]
            lengths = [len(_) for _ in sorted_lines]        
            orders = [i for i in range(len(sorted_lines))]
            
            sorted_tuples = sorted(zip(sorted_lines, lengths, orders), key = itemgetter(1), reverse = True)
            sorted_lines = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            orders = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # Converts string to list of index.
            x = loader.src.numericalize(loader.src.pad(sorted_lines), device = 'cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu')

            if config.beam_size == 1:
                # Take inference for non-parallel beam-search.
                y_hat, indice = model.search(x)
                output = to_text(indice, loader.tgt.vocab)

                sorted_tuples = sorted(zip(output, orders), key = itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(x, 
                                                            beam_size = config.beam_size, 
                                                            max_length = config.max_length, 
                                                            n_best = config.n_best
                                                            )

                # Restore the original orders.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, orders), key = itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
```

## data_loader.py

```python
import os
from torchtext import data, datasets

PAD = 1
BOS = 2
EOS = 3

class DataLoader():

    def __init__(self, train_fn = None, 
                    valid_fn = None, 
                    exts = None,
                    batch_size = 64, 
                    device = 'cpu', 
                    max_vocab = 99999999, 
                    max_length = 255, 
                    fix_length = None, 
                    use_bos = True, 
                    use_eos = True, 
                    shuffle = True
                    ):

        super(DataLoader, self).__init__()

        self.src = data.Field(sequential = True, 
                                    use_vocab = True, 
                                    batch_first = True, 
                                    include_lengths = True, 
                                    fix_length = fix_length, 
                                    init_token = None, 
                                    eos_token = None
                                    )

        self.tgt = data.Field(sequential = True, 
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = '<BOS>' if use_bos else None, 
                                eos_token = '<EOS>' if use_eos else None
                                )

        if train_fn is not None and valid_fn is not None and exts is not None:
            train = TranslationDataset(path = train_fn, exts = exts,
                                            fields = [('src', self.src), ('tgt', self.tgt)], 
                                            max_length = max_length
                                            )
            valid = TranslationDataset(path = valid_fn, exts = exts,
                                            fields = [('src', self.src), ('tgt', self.tgt)], 
                                            max_length = max_length
                                            )

            self.train_iter = data.BucketIterator(train, 
                                                    batch_size = batch_size, 
                                                    device = 'cuda:%d' % device if device >= 0 else 'cpu', 
                                                    shuffle = shuffle, 
                                                    sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), 
                                                    sort_within_batch = True
                                                    )
            self.valid_iter = data.BucketIterator(valid, 
                                                    batch_size = batch_size, 
                                                    device = 'cuda:%d' % device if device >= 0 else 'cpu', 
                                                    shuffle = False, 
                                                    sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), 
                                                    sort_within_batch = True
                                                    )

            self.src.build_vocab(train, max_size = max_vocab)
            self.tgt.build_vocab(train, max_size = max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab

class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        if not path.endswith('.'):
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1], sys.argv[2], (sys.argv[3], sys.argv[4]), batch_size = 8)

    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)

        if batch_index > 1:
            break
```

## simple_nmt/seq2seq.py

```python
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_loader
from simple_nmt.search import SingleBeamSearchSpace

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, h_src, h_t_tgt, mask = None):
        # |h_src| = (batch_size, length, hidden_size)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length)

        query = self.linear(h_t_tgt.squeeze(1)).unsqueeze(-1)
        # |query| = (batch_size, hidden_size, 1)

        weight = torch.bmm(h_src, query).squeeze(-1)
        # |weight| = (batch_size, length)
        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weight.masked_fill_(mask, -float('inf'))
        weight = self.softmax(weight)

        context_vector = torch.bmm(weight.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        return context_vector

class Encoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers = 4, dropout_p = .2):
        super(Encoder, self).__init__()

        # Be aware of value of 'batch_first' parameter.
        # Also, its hidden_size is half of original hidden_size, because it is bidirectional.
        self.rnn = nn.LSTM(word_vec_dim, int(hidden_size / 2), num_layers = n_layers, dropout = dropout_p, bidirectional = True, batch_first = True)

    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_dim)

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first = True)

            # Below is how pack_padded_sequence works.
            # As you can see, PackedSequence object has information about mini-batch-wise information, not time-step-wise information.
            # 
            # a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            # >>>>
            # tensor([[ 1,  2,  3],
            #     [ 3,  4,  0]])
            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
            # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
        else:
            x = emb
        
        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first = True)

        return y, h

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

class Generator(nn.Module):
    
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)

        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)

        # Return log-probability instead of just probability.
        return y

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

## simple_nmt/trainer.py

```python
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils

def get_loss(y, y_hat, criterion, do_backward = True):
    # |y| = (batch_size, length)
    # |y_hat| = (batch_size, length, output_size)
    batch_size = y.size(0)

    loss = criterion(y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1))
    if do_backward:
        loss.div(batch_size).backward()

    return loss

def train_epoch(model, criterion, train_iter, valid_iter, config, start_epoch = 1, others_to_save = None):
    current_lr = config.lr

    lowest_valid_loss = np.inf
    no_improve_cnt = 0

    for epoch in range(start_epoch, config.n_epochs + 1):
        if config.adam:
            optimizer = optim.Adam(model.parameters(), lr = current_lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr = current_lr)
        print("current learning rate: %f" % current_lr)
        print(optimizer)

        sample_cnt = 0
        total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
        start_time = time.time()
        train_loss = np.inf

        for batch_index, batch in enumerate(train_iter):
            # You have to reset the gradients of all model parameters before to take another step in gradient descent.
            optimizer.zero_grad()

            current_batch_word_cnt = torch.sum(batch.tgt[1])
            x = batch.src
            # Raw target variable has both BOS and EOS token. 
            # The output of sequence-to-sequence does not have BOS token. 
            # Thus, remove BOS token for reference.
            y = batch.tgt[0][:, 1:] 
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = model(x, batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            # Calcuate loss and gradients with back-propagation.
            loss = get_loss(y, y_hat, criterion)
            
            # Simple math to show stats.
            total_loss += float(loss)
            total_word_count += int(current_batch_word_cnt)
            total_parameter_norm += float(utils.get_parameter_norm(model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(model.parameters()))

            # Print current training status in every this number of mini-batch is done.
            if (batch_index + 1) % config.print_every == 0:
                avg_loss = total_loss / total_word_count
                avg_parameter_norm = total_parameter_norm / config.print_every
                avg_grad_norm = total_grad_norm / config.print_every
                elapsed_time = time.time() - start_time

                # You can check the current status using parameter norm and gradient norm.
                # Also, you can check the speed of the training.
                print("epoch: %d batch: %d/%d\t|param|: %.2f\t|g_param|: %.2f\tloss: %.4f\tPPL: %.2f\t%5d words/s %3d secs" % (epoch, 
                                                                                                            batch_index + 1, 
                                                                                                            int(len(train_iter.dataset.examples) // config.batch_size), 
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

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += batch.tgt[0].size(0)
            if sample_cnt >= len(train_iter.dataset.examples):
                break

        sample_cnt = 0
        total_loss, total_word_count = 0, 0

        with torch.no_grad(): # In validation, we don't need to get gradients.
            model.eval() # Turn-on the evaluation mode.
            
            for batch_index, batch in enumerate(valid_iter):
                current_batch_word_cnt = torch.sum(batch.tgt[1])
                x = batch.src
                y = batch.tgt[0][:, 1:]
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)

                # Take feed-forward
                y_hat = model(x, batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, length, output_size)

                loss = get_loss(y, y_hat, criterion, do_backward = False)

                total_loss += float(loss)
                total_word_count += int(current_batch_word_cnt)

                sample_cnt += batch.tgt[0].size(0)
                if sample_cnt >= len(valid_iter.dataset.examples):
                    break

            # Print result of validation.
            avg_loss = total_loss / total_word_count
            print("valid loss: %.4f\tPPL: %.2f" % (avg_loss, np.exp(avg_loss)))

            if lowest_valid_loss > avg_loss:
                lowest_valid_loss = avg_loss
                no_improve_cnt = 0

                # Altough there is an improvement in last epoch, we need to decay the learning-rate if it meets the requirements.
                if epoch >= config.lr_decay_start_at:
                    current_lr = max(config.min_lr, current_lr * config.lr_decay_rate)
            else:
                # Decrease learing rate if there is no improvement.
                current_lr = max(config.min_lr, current_lr * config.lr_decay_rate)
                no_improve_cnt += 1

            # Again, turn-on the training mode.
            model.train()

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model.split(".")
        model_fn = model_fn[:-1] + ["%02d" % epoch, "%.2f-%.2f" % (train_loss, np.exp(train_loss)), "%.2f-%.2f" % (avg_loss, np.exp(avg_loss))] + [model_fn[-1]]

        # PyTorch provides efficient method for save and load model, which uses python pickle.
        to_save = {"model": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "current_lr": current_lr
                    }
        if others_to_save is not None: # Add others if it is necessary.
            for k, v in others_to_save.items():
                to_save[k] = v
        torch.save(to_save, '.'.join(model_fn))

        # Take early stopping if it meets the requirement.
        if config.early_stop > 0 and no_improve_cnt > config.early_stop:
            break
```

## simple_nmt/search.py

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

## utils.py

```python

import torch

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