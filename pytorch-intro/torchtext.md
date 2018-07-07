# TorchText

## 소개

[TorchText Docs](http://torchtext.readthedocs.io/en/latest/index.html#)

## How to install

```bash
$ pip install torchtext
```

## Example

### Reading Monolingual Corpus

```python
from torchtext import data, datasets

PAD = 1
BOS = 2
EOS = 3

class DataLoader():

    def __init__(self, train_fn, valid_fn, 
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

        self.text = data.Field(sequential = True, 
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = '<BOS>' if use_bos else None, 
                                eos_token = '<EOS>' if use_eos else None
                                )

        train = LanguageModelDataset(path = train_fn, 
                                        fields = [('text', self.text)], 
                                        max_length = max_length
                                        )
        valid = LanguageModelDataset(path = valid_fn, 
                                        fields = [('text', self.text)], 
                                        max_length = max_length
                                        )

        self.train_iter = data.BucketIterator(train, 
                                                batch_size = batch_size, 
                                                device = 'cuda:%d' % device if device >= 0 else 'cpu', 
                                                shuffle = shuffle, 
                                                sort_key=lambda x: -len(x.text), 
                                                sort_within_batch = True
                                                )
        self.valid_iter = data.BucketIterator(valid, 
                                                batch_size = batch_size, 
                                                device = 'cuda:%d' % device if device >= 0 else 'cpu', 
                                                shuffle = False, 
                                                sort_key=lambda x: -len(x.text), 
                                                sort_within_batch = True
                                                )

        self.text.build_vocab(train, max_size = max_vocab)

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

```python
if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1], sys.argv[2])

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.text)

        if batch_index >= 1:
            break
```

### Reading Monolingual Corpus with Labeling

```python
class TextClassificationDataLoader():

    def __init__(self, train_fn, valid_fn, tokenizer, 
                                            batch_size = 64, 
                                            device = 'cpu', 
                                            max_vocab = 9999999, 
                                            fix_length = None, 
                                            use_eos = False, 
                                            shuffle = True):
        
        super(TextClassificationDataLoader, self).__init__()

        self.label = data.Field(sequential = False, use_vocab = False)
        self.text = data.Field(tokenize = tokenizer, 
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                eos_token = '<EOS>' if use_eos else None
                                )

        train, valid = data.TabularDataset.splits(path = '', 
                                                    train = train_fn, 
                                                    validation = valid_fn, 
                                                    format = 'tsv', 
                                                    fields = [('label', self.label), ('text', self.text)]
                                                    )

        self.train_iter, self.valid_iter = data.BucketIterator.splits((train, valid), 
                                                                        batch_size = batch_size, 
                                                                        device = 'cuda:%d' % device if device >= 0 else 'cpu', 
                                                                        shuffle = shuffle
                                                                        )
        
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size = max_vocab)
```

### Reading Parallel(Bi-lingual) Corpus

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
```