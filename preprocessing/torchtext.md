# TorchText(토치텍스트)

사실 딥러닝 코드를 작성하다 보면, 신경망 모델 자체를 코딩하는 시간보다 그 모델을 훈련하도록 하는 코드를 짜는 시간이 더 오래걸리기 마련입니다. 데이터 입력을 준비하는 부분도 이에 해당 합니다. TorchText는 자연어처리 문제 또는 텍스트와 관련된 기계학습 또는 딥러닝을 수행하기 위한 데이터를 읽고 전처리 하는 코드를 모아놓은 라이브러리 입니다.

이번 섹션에서는 TorchText를 활용한 기본적인 데이터 로딩 방법에 대한 실습을 해 보도록 하겠습니다. 좀 더 자세한 내용은 [TorchText 문서](http://torchtext.readthedocs.io/en/latest/index.html#)를 참조 해 주세요.

## 설치 방법

TorchText는 pip을 통해서 쉽게 설치 가능 합니다. 아래와 같이 명령어를 수행하여 설치 합니다.

```bash
$ pip install torchtext
```

## 예제

사실 자연어처리 분야에서 주로 사용되는 학습 데이터의 형태는 크게 3가지로 분류할 수 있습니다. 주로 우리의 신경망이 하고자 하는 바는 입력 $x$를 받아 알맞은 출력 $y$를 반환해 주는 함수의 형태이므로, $x, y$의 형태에 따라서 분류 해 볼 수 있습니다.

| x 데이터 | y 데이터 | 어플리케이션 |
| --- | --- | --- |
| corpus | 클래스(class) | 텍스트 분류(text classification), 감성분석(sentiment analysis) |
| corpus | - | 언어모델(language modeling) |
| corpus | corpus | 기계번역(machine translation), 요약(summarization), QnA |

따라서 우리는 이 3가지 종류의 데이터 형태를 다루는 방법에 대해서 실습합니다. 사실 TorchText는 훨씬 더 복잡하고 정교한 함수들을 제공합니다. 하지만 저자가 느끼기에는 좀 과도하고 직관적이지 않은 부분들이 많아, 제공되는 함수들의 사용을 최소화 하여 복잡하지 않고 간단한 방식으로 구현해보고자 합니다.

### 코퍼스와 레이블(label)을 읽기

첫번째 예제는 한 줄 안에서 클래스(class)와 텍스트(text)가 tab(탭, '\\t')으로 구분되어 있는 데이터의 입력을 받기 위한 예제 입니다. 주로 이런 예제는 텍스트 분류(text classification)을 위해 사용 됩니다.

```python
class TextClassificationDataLoader():

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

### 코퍼스 읽기

이 예제는 한 라인에 텍스트로만 채워져 있을 때를 위한 코드 입니다. 주로 언어모델(language model)을 훈련 시키는 상황에서 사용 될 수 있습니다.

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

### 병렬(또는 쌍으로 된) 코퍼스 읽기

아래의 코드는 텍스트로만 채워진 두개의 파일을 동시에 입력 데이터로 읽어들이는 예제 입니다. 이때, 두 파일의 코퍼스는 병렬(parallel) 데이터로 취급되어 같은 라인 수로 채워져 있어야 합니다. 주로 기계번역(machine translation)이나 요약(summarization) 등에 사용 할 수 있습니다.

```python
from torchtext import data, datasets

PAD, BOS, EOS = 1, 2, 3


class DataLoader():

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
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

        self.src = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token=None,
                              eos_token=None
                              )

        self.tgt = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if use_bos else None,
                              eos_token='<EOS>' if use_eos else None
                              )

        if train_fn is not None and valid_fn is not None and exts is not None:
            train = TranslationDataset(path=train_fn,
                                       exts=exts,
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )
            valid = TranslationDataset(path=valid_fn,
                                       exts=exts,
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )

            self.train_iter = data.BucketIterator(train,
                                                  batch_size=batch_size,
                                                  device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  shuffle=shuffle,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True
                                                  )
            self.valid_iter = data.BucketIterator(valid,
                                                  batch_size=batch_size,
                                                  device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  shuffle=False,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True
                                                  )

            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab


class TranslationDataset(data.Dataset):

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
                if max_length and max_length < max(len(src_line.split()),
                                                   len(trg_line.split())
                                                   ):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)
```