# n-gram Exercise with SRILM

SRILM은 음성인식, segmentation, 기계번역 등에 사용되는 통계 언어 모델 (n-gram language model)을 구축하고 적용 할 수 있는 toolkit입니다. 이 책에서 다루는 다른 알고리즘이나 기법들에 비하면, SRI speech research lab에서 1995년부터 연구/개발 해 온 유서깊은(?) toolkit 입니다.

## Install SRILM

> 다운로드: http://www.speech.sri.com/projects/srilm/download.html

위의 주소에서 SRILM은 간단한 정보를 기입 한 후, 다운로드 가능합니다. 이후에 아래와 같이 디렉터리를 만들고 그 안에 압축을 풀어 놓습니다.

```bash
$ mkdir srilm
$ cd ./srilm
$ tar –xzvf ./srilm-1.7.2.tar.gz
```

디렉터리 내부에 Makefile을 열어 7번째 라인의 SRILM의 경로 지정 후에 주석을 해제 하여 줍니다. 그리고 make명령을 통해 SRILM을 빌드 합니다.

```bash
$ vi ./Makefile
$ make
```

빌드가 정상적으로 완료 된 후에, PATH에 SRILM/bin 내부에 새롭게 생성된 디렉터리를 등록 한 후, export 해 줍니다.

```
PATH={SRILM_PATH}/bin/{MACHINE}:$PATH
#PATH=/home/khkim/Workspace/nlp/srilm/bin/i686-m64:$PATH
export PATH
```

그리고 아래와 같이 ngram-count와 ngram이 정상적으로 동작하는 것을 확인 합니다.

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