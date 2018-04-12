# Align Parallel Corpus

대부분의 parallel corpora는 여러 문장 단위로 align이 되어 있는 경우가 많습니다. 이러한 경우에는 한 문장씩에 대해서 align을 해주어야 합니다. 또한, 이러한 과정에서 일부 parallel 하지 않은 문장들을 걸러내야 하고, 문장 간 align이 잘 맞지 않는 경우 align을 재정비 해 주거나 아예 걸러내야 합니다. 이러한 과정에 대해서 살펴 봅니다.

## Process Overview for Parallel Corpus Alignment

1. Building a dictionary between source language to target language.
    1. Collect and normalize (clean + tokenize) corpus for each language.
    1. Get word embedding vector for each language.
    1. Get word-level-translator using [MUSE](https://github.com/facebookresearch/MUSE).
1. Align collected semi-parallel corpus using [Champollion](https://github.com/LowResourceLanguages/champollion).
    1. Sentence-tokenize for each language.
    1. Normalize (clean + tokenize) corpus for each language.
    1. Align parallel corpus using Champollion.

## Building Dictionary

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