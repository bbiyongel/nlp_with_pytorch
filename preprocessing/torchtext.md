# 토치텍스트(TorchText)

사실 딥러닝 코드를 작성하다 보면, 신경망 모델 자체를 코딩하는 시간보다 그 모델을 훈련하도록 하는 코드를 짜는 시간이 더 오래걸리기 마련입니다. 데이터 입력을 준비하는 부분도 이에 해당 합니다. 토치텍스트는 자연어처리 문제 또는 텍스트와 관련된 기계학습 또는 딥러닝을 수행하기 위한 데이터를 읽고 전처리 하는 코드를 모아놓은 라이브러리 입니다. 우리가 앞으로 설명할 텍스트 분류나 언어모델, 그리고 기계번역의 경우에도 토치텍스트를 활용하여 쉽게 텍스트 파일을 읽어 훈련에 사용합니다.

이번 섹션에서는 토치텍스트를 활용한 기본적인 데이터 로딩 방법에 대한 실습을 해 보도록 하겠습니다. 좀 더 자세한 내용은 [토치텍스트 문서](http://torchtext.readthedocs.io/en/latest/index.html#)를 참조 해 주세요.

## 설치 방법

토치텍스트는 pip을 통해서 쉽게 설치 가능 합니다. 아래와 같이 명령어를 수행하여 설치 합니다.

```bash
$ pip install torchtext
```

## 예제 코드

사실 자연어처리 분야에서 주로 사용되는 학습 데이터의 형태는 크게 3가지로 분류할 수 있습니다. 주로 우리의 신경망이 하고자 하는 바는 입력 $x$ 를 받아 알맞은 출력 $y$ 를 반환해 주는 함수의 형태이므로, $x, y$ 의 형태에 따라서 분류 해 볼 수 있습니다.

| x 데이터 | y 데이터 | 활용분야 |
|-|-|-|
| 코퍼스 | 클래스(class) | 텍스트 분류(text classification), 감성분석(sentiment analysis) |
| 코퍼스 | - | 언어모델(language modeling) |
| 코퍼스 | 코퍼스 | 기계번역(machine translation), 요약(summarization), QnA |

따라서 우리는 이 3가지 종류의 데이터 형태를 다루는 방법에 대해서 실습합니다. 사실 토치텍스트는 훨씬 더 복잡하고 정교한 함수들을 제공합니다. 하지만 저자가 느끼기에는 좀 과도하고 직관적이지 않은 부분들이 많아, 제공되는 함수들의 사용을 최소화 하여 복잡하지 않고 간단한 방식으로 구현해보고자 합니다.

주로 'Field'라는 클래스를 통해 먼저 우리가 읽고자 하는 텍스트 파일 내의 필드를 정의 합니다. 탭(tab)을 통해 필드를 구분하는 방식을 자연어처리 분야의 입력에서 가장 많이 사용합니다. 콤마(comma)의 경우에는 텍스트 내부에 콤마가 포함될 가능성이 많기 때문에, 콤마를 딜리미터(delimiter)로 사용하는 것은 위험한 선택이 될 가능성이 높습니다. 이렇게 정의된 각 필드를 'Dataset' 클래스를 통해 읽어들입니다. 이렇게 읽어들인 코퍼스는 미리 주어진 미니배치 사이즈에 따라서 나뉘게 iterator에 담기게 됩니다. 미니배치를 구성하는 과정에서 미니배치 내에 문장의 길이가 다를 경우에는 정책에 따라서 문장의 앞 또는 뒤에 패딩(padding, PAD)을 삽입합니다. 이 패딩은 추후 소개할 BOS, EOS와 함께 하나의 단어 또는 토큰과 같은 취급을 받게 됩니다. 이후에 훈련 코퍼스에 대해서 어휘 사전을 만들어 각 단어(토큰)을 숫자로 맵핑하는 작업을 수행하면 됩니다.

### 코퍼스와 레이블(label)을 읽기

첫번째 예제는 한 줄 안에서 클래스(class)와 텍스트(text)가 tab(탭, '\\t')으로 구분되어 있는 데이터의 입력을 받기 위한 예제 입니다. 주로 이런 예제는 텍스트 분류(text classification)을 위해 사용 됩니다.

- URL: https://github.com/kh-kim/simple-ntc/blob/master/data_loader.py

```python
from torchtext import data


class DataLoader(object):

    def __init__(self, train_fn, valid_fn, 
                 batch_size=64, 
                 device=-1, 
                 max_vocab=999999, 
                 min_freq=1,
                 use_eos=False, 
                 shuffle=True
                 ):
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

이 예제는 한 라인에 텍스트로만 채워져 있을 때를 위한 코드 입니다. 주로 언어모델(language model)을 훈련 시키는 상황에서 사용 될 수 있습니다. 'LanguageModelDataset'을 통해서 미리 정의된 필드를 텍스트 파일에서 읽어들입니다. 이때, 각 문장의 길이에 따라서 정렬을 통해 비슷한 길이의 문장끼리 미니배치를 만들어주는 작업을 수행합니다. 이 작업을 통해서 추후 매우 상이한 길이의 문장들이 하나의 미니배치에 묶여 훈련 시간에서 손해보는 것을 방지합니다.

- URL: https://github.com/kh-kim/OpenNLMTK/blob/master/data_loader.py

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

아래의 코드는 텍스트로만 채워진 두개의 파일을 동시에 입력 데이터로 읽어들이는 예제 입니다. 이때, 두 파일의 코퍼스는 병렬(parallel) 데이터로 취급되어 같은 라인 수로 채워져 있어야 합니다. 주로 기계번역(machine translation)이나 요약(summarization) 등에 사용 할 수 있습니다. 탭(tab)을 사용하여 하나의 파일에서 두 개의 컬럼에 각 언어의 문장을 표현하는 것도 한가지 방법이 될 수 있습니다. 그렇다면 앞서 소개한 'TabularDataset' 클래스를 이용하면 될 것 입니다. 그리고 앞서 소개한 'LanguageModelDataset'과 마찬가지로 길이에 따라서 미니배치를 구성하도록 합니다.

- URL: https://github.com/kh-kim/simple-nmt/blob/master/data_loader.py

```python
import os
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
                 shuffle=True,
                 dsl=False
                 ):

        super(DataLoader, self).__init__()

        self.src = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if dsl else None,
                              eos_token='<EOS>' if dsl else None
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
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        if not path.endswith('.'):
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()),
                                                   len(trg_line.split())
                                                   ):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super().__init__(examples, fields, **kwargs)
```