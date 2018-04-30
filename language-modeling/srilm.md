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

이전 preprocessing 챕터에서 다루었던 대로 tokenize가 완료된 파일을 데이터로 사용합니다. 그렇게 준비된 파일을 training set과 test set으로 나누어 준비 합니다.

## Basic Usage

아래는 주로 SRILM에서 사용되는 프로그램들의 주요 arguments에 대한 설명입니다.

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

위에서 나온 대로, language model을 훈련하기 위해서는 ngram-count 모듈을 이용 합니다. 아래의 명령어는 예를 들어 kndiscount를 사용한 상태에서 tri-gram을 훈련하고 LM과 거기에 사용된 vocabulary를 출력하도록 하는 명령 입니다.

```bash
$ time ngram-count -order 3 -kndiscount -text <text_fn> -lm <output_lm_fn> -write_vocab <output_vocab_fn> -debug 2
```

### Sentence Generation

아래의 명령은 위와 같이 ngram-count 모듈을 사용해서 만들어진 lm을 활용하여 문장을 생성하는 명령입니다. 문장을 생성 한 이후에는 preprocessing 챕터에서 다루었듯이 detokenization을 수행해 주어야 하며, pipeline을 통해 sed와 regular expression을 사용하여 detokenization을 해 주도록 해 주었습니다.

```bash
$ ngram -lm <input_lm_fn> -gen <n_sentence_to_generate> | sed "s/ //g" | sed "s/▁▁/ /g" | sed "s/▁//g" | sed "s/^\s//g"
```

위와 같이 매번 sed와 regular expression을 통하는 것이 번거롭다면, preprocessing 챕터에서 제공되었던 detokenization python script를 통하여 detokenization을 수행 할 수도 있습니다.

### Evaluation

```bash
$ ngram -ppl <test_fn> -lm <input_lm_fn> -order 3 -debug 2
```

```bash
$ ngram -ppl ./data/test.refined.tok.bpe.txt -lm ./data/ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2
file ./data/test.refined.tok.bpe.txt: 1000 sentences, 13302 words, 32 OOVs0 zeroprobs, logprob= -36717.49 ppl= 374.1577 ppl1= 584.7292
```

### Interpolation

```bash
$ ngram -lm <input_lm_fn> -mix-lm <mix_lm_fn> -lambda <mix_ratio_between_0_and_1> -write-lm <output_lm_fn> -debug 2
```