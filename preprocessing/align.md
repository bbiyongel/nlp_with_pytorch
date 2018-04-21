# Align Parallel Corpus

대부분의 parallel corpora는 여러 문장 단위로 align이 되어 있는 경우가 많습니다. 이러한 경우에는 한 문장씩에 대해서 align을 해주어야 합니다. 또한, 이러한 과정에서 일부 parallel 하지 않은 문장들을 걸러내야 하고, 문장 간 align이 잘 맞지 않는 경우 align을 재정비 해 주거나 아예 걸러내야 합니다. 이러한 과정에 대해서 살펴 봅니다.

## Process Overview for Parallel Corpus Alignment

Alignment를 수행하기 위한 전체 과정을 요약하면 아래와 같습니다.

1. Building a dictionary between source language to target language.
    1. Collect and normalize (clean + tokenize) corpus for each language.
    1. Get word embedding vector for each language.
    1. Get word-level-translator using [MUSE](https://github.com/facebookresearch/MUSE).
1. Align collected semi-parallel corpus using [Champollion](https://github.com/LowResourceLanguages/champollion).
    1. Sentence-tokenize for each language.
    1. Normalize (clean + tokenize) corpus for each language.
    1. Align parallel corpus using Champollion.

## Building Dictionary

만약 기존에 단어 (번역) 사전을 구축해 놓았다면 그것을 이용하면 되지만, 단어 사전을 구축하는 것 또한 비용이 들기 때문에, 일반적으로는 쉽지 않습니다. 따라서, 단어 사전을 구축하는 것 또한 자동으로 할 수 있습니다.

Facebook의 [MUSE](https://github.com/facebookresearch/MUSE)는 parallel corpora가 없는 상황에서 사전을 구축하는 방법을 제시하고, 코드를 제공합니다. 각 Monolingual corpus를 통해 구축한 언어 별 word embedding vector에 대해서 다른 언어의 embedding vector와 mapping하도록 함으로써, 단어 간 번역을 할 수 있습니다. 이는 각 언어 별 corpus가 많을 수록 embedding vector가 많을수록 더욱 정확하게 수행 됩니다. MUSE는 parallel corpora가 없는 상황에서도 수행 할 수 있기 때문에 unsupervised learning이라고 할 수 있습니다.

아래는 MUSE를 통해 구한 영한 사전의 일부로써, 꽤 정확한 단어 간의 번역을 볼 수 있습니다. 이렇게 구성한 사전은 champollion의 입력으로 사용되어, champollion은 이 사전을 바탕으로 parallel corpus의 sentence alignment를 수행합니다. **<>**을 delimiter로 사용하여 한 라인에 source 언어의 단어와 target 언어의 단어를 표현 합니다.

```
stories <> 이야기
stories <> 소설
contact <> 연락
contact <> 연락처
contact <> 접촉
green <> 녹색
green <> 초록색
green <> 빨간색
dark <> 어두운
dark <> 어둠
dark <> 짙
song <> 노래
song <> 곡
song <> 음악
salt <> 소금
```

## Align via Champollion

Chapollion Toolkit(CTK)는 parallel corpus의 sentence alignement를 수행하는 open-source입니다. Perl을 사용하여 구현되었으며, 이집트 상형문자를 처음으로 해독해낸 역사학자 [Champollion](https://ko.wikipedia.org/wiki/%EC%9E%A5%ED%94%84%EB%9E%91%EC%88%98%EC%95%84_%EC%83%B9%ED%8F%B4%EB%A6%AC%EC%98%B9)의 이름을 따서 명명되었습니다.

![](https://upload.wikimedia.org/wikipedia/commons/f/ff/Jean-Francois_Champollion.jpg)<br>
(Jean-François Champollion, Image from Wikipedia)

기-구축된 단어 (번역) 사전을 이용하거나, 위와 같이 자동으로 구축한 단어 사전을 참고하여 Champollion은 sentence alignment를 수행합니다. 여러 line으로 구성된 각 언어 별 하나의 document에 대해서 sentence alignment를 수행한 결과는 아래와 같습니다.

```
omitted <=> 1
omitted <=> 2
omitted <=> 3
1 <=> 4
2 <=> 5
3 <=> 6
4,5 <=> 7
6 <=> 8
7 <=> 9
8 <=> 10
9 <=> omitted
```

위의 결과를 해석해 보면, target 언어의 1, 2, 3번째 문장은 짝을 찾지 못하고 버려졌고, source 언어의 1, 2, 3번째 문장은 각각 target 언어의 4, 5, 6번째 문장과 mapping 된 것을 알 수 있습니다. 또한, source 언어의 4, 5번째 두 문장은 target 언어의 7번 문장에 동시에 mapping 된 것을 볼 수 있습니다.

이와 같이 어떤 문장들은 버려지기도 하고, 일대일 mapping이 이루어지기도 하며, 일대다, 다대일 mapping이 이루어지기도 합니다.

아래는 champollion을 쉽게 사용하기 위한 Python script 예제입니다. ***CTK_ROOT***에 Chapollion Toolkit의 위치를 지정하여 사용할 수 있습니다.

```python
import sys, argparse, os

BIN = CTK_ROOT + "/bin/champollion"
CMD = "%s -c %f -d %s %s %s %s"
OMIT = "omitted"
INTERMEDIATE_FN = "./tmp/tmp.txt"

def read_alignment(fn):
    aligns = []

    f = open(fn, 'r')

    for line in f:
        if line.strip() != "":
            srcs, tgts = line.strip().split(' <=> ')

            if srcs == OMIT:
                srcs = []
            else:
                srcs = list(map(int, srcs.split(',')))

            if tgts == OMIT:
                tgts = []
            else:
                tgts = list(map(int, tgts.split(',')))

            aligns += [(srcs, tgts)]

    f.close()

    return aligns

def get_aligned_corpus(src_fn, tgt_fn, aligns):
    f_src = open(src_fn, 'r')
    f_tgt = open(tgt_fn, 'r')

    for align in aligns:
        srcs, tgts = align

        src_buf, tgt_buf = [], []

        for src in srcs:
            src_buf += [f_src.readline().strip()]
        for tgt in tgts:
            tgt_buf += [f_tgt.readline().strip()]

        if len(src_buf) > 0 and len(tgt_buf) > 0:
            sys.stdout.write("%s\t%s\n" % (" ".join(src_buf), " ".join(tgt_buf)))

    f_tgt.close()
    f_src.close()

def parse_argument():
    p = argparse.ArgumentParser()

    p.add_argument('--src', required = True)
    p.add_argument('--tgt', required = True)
    p.add_argument('--src_ref', default = None)
    p.add_argument('--tgt_ref', default = None)
    p.add_argument('--dict', required = True)
    p.add_argument('--ratio', type = float, default = 1.2750)

    config = p.parse_args()

    return config

if __name__ == "__main__":
    config = parse_argument()

    if config.src_ref is None:
        config.src_ref = config.src
    if config.tgt_ref is None:
        config.tgt_ref = config.tgt

    cmd = CMD % (BIN, config.ratio, config.dict, config.src_ref, config.tgt_ref, INTERMEDIATE_FN)
    os.system(cmd)

    aligns = read_alignment(INTERMEDIATE_FN)
    get_aligned_corpus(config.src, config.tgt, aligns)

```

특기할 점은 ratio parameter의 역할입니다. 이 parameter는 실제 champollion의 **-c** 옵션으로 mapping되어 사용되는데, champollion 상에서의 설명은 다음과 같습니다.

```bash
$ ./champollion
usage: ./champollion [-hdscn] <X token file> <Y token file> <alignment file>

      -h       : this (help) message
      -d dictf : use dictf as the translation dictionary
      -s xstop : use words in file xstop as X stop words
      -c n     : number of Y chars for each X char
      -n       : disallow 1-3, 3-1, 1-4, 4-1 alignments
              (faster, lower performance)
```

즉, source language의 charactor 당 target language의 charactor 비율을 의미합니다. 이를 기반하여 champollion은 문장 내 모든 단어에 대해서 번역 단어를 모르더라도 sentence alignment를 수행할 수 있게 됩니다.