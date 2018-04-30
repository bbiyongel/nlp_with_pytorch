# n-gram Exercise with SRILM

SRILM은 음성인식, segmentation, 기계번역 등에 사용되는 통계 언어 모델 (n-gram language model)을 구축하고 적용 할 수 있는 toolkit입니다. 이 책에서 다루는 다른 알고리즘이나 기법들에 비하면, SRI speech research lab에서 1995년부터 연구/개발 해 온 유서깊은(?) toolkit 입니다.

## Install SRILM

> http://www.speech.sri.com/projects/srilm/download.html

```bash
$ mkdir srilm
$ cd ./srilm
$ tar –xzvf ./srilm-1.7.2.tar.gz
```
```bash
$ vi ./Makefile
```
7번째 라인 # SRILM = ‘@#$%@#$” 을 경로 지정 후 de-commentize

```bash
$ make
```
```
PATH={SRILM_PATH}/bin/{MACHINE}:$PATH
# PATH=/home/khkim/Workspace/nlp/srilm/bin/i686-m64:$PATH
export PATH
```
```bash
$ source ~/.profile
$ ngram-count -help
$ ngram -help
```

## Prepare Dataset

## Basic Usage

- ngram-count: LM을 훈련
    - vocab: lexicon file name
    - text: training corpus file name 
    - order: n-gram count 
    - write: output countfile file name 
    - unk: mark OOV as
    - kndiscountn: Use Kneser-Ney discounting for N-grams of order n


- ngram: LM을 활용
    - ppl: calculate perplexity for test file name
    - order: n-gram count
    - lm: language model file name

### Language Modeling

```bash
$ time ngram-count -order 3 -kndiscount -text <text_fn> -lm <output_lm_fn> -write_vocab <output_vocab_fn> -debug 2
```

### Sentence Generation

```bash
$ ngram -lm <input_lm_fn> -gen <n_sentence_to_generate> | python {PREPROC_PATH}/detokenizer.py
```

### Evaluation

```bash
$ ngram -ppl <test_fn> -lm <input_lm_fn> -order 3 -debug 2
```

### Interpolation

```bash
$ ngram -lm <input_lm_fn> -mix-lm <mix_lm_fn> -lambda <mix_ratio_between_0_and_1> -write-lm <output_lm_fn> -debug 2
```