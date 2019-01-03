# SRILM을 활용하여 n-gram 실습하기

SRILM은 음성인식, 분절, 기계번역 등에 사용되는 통계 언어 모델 (n-gram language model)을 구축하고 적용 할 수 있는 툴킷 입니다. 이 책에서 다루는 다른 알고리즘이나 기법들에 비하면, SRI speech research lab에서 1995년부터 연구/개발해 온 유서깊은 툴킷 입니다.

## SRILM 설치

- 다운로드: http://www.speech.sri.com/projects/srilm/download.html

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

## 데이터셋 준비하기

이전 전처리 챕터에서 다루었던 대로 분절이 완료된 파일을 데이터로 사용합니다. 그렇게 준비된 파일을 훈련 데이터와 테스트 데이터로 나누어 준비 합니다.

## 기본 사용법

아래는 주로 SRILM에서 사용되는 프로그램들의 주요 인자에 대한 설명입니다.

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

### 언어모델 만들기

위에서 나온 대로, 언어모델을 훈련하기 위해서는 ngram-count 모듈을 이용 합니다. 아래의 명령어는 예를 들어 kndiscount를 사용한 상태에서 3-gram을 훈련하고 언어모델과 언어모델을 구성하는 어휘 사전을 출력하도록 하는 명령 입니다.

```bash
$ time ngram-count -order 3 -kndiscount -text <text_fn> -lm <output_lm_fn> -write_vocab <output_vocab_fn> -debug 2
```

### 문장 생성하기

아래의 명령은 위와 같이 ngram-count 모듈을 사용해서 만들어진 언어모델을 활용하여 문장을 생성하는 명령입니다. 문장을 생성 한 이후에는 전처리 챕터에서 다루었듯이 분절을 복원하는 작업(detokenization)을 수행해 주어야 하며, 리눅스 pipeline을 연계하여 sed를 통한 regular expression을 사용하여 분절을 복원 하도록 하였습니다.

```bash
$ ngram -lm <input_lm_fn> -gen <n_sentence_to_generate> | sed "s/ //g" | sed "s/▁▁/ /g" | sed "s/▁//g" | sed "s/^\s//g"
```

위와 같이 매번 sed와 regular expression을 통하는 것이 번거롭다면, 파이썬으로 직접 해당 작업을 하도록 코드를 작성하는 것도 좋은 방법 입니다.

### 평가

이렇게 언어모델을 훈련하고 나면 테스트 데이터에 대해서 평가를 통해 얼마나 훌륭한 언어모델이 만들어졌는지 평가할 필요가 있습니다. 언어모델에 대한 성능평가는 아래와 같은 명령을 통해 수행 될 수 있습니다.

```bash
$ ngram -ppl <test_fn> -lm <input_lm_fn> -order 3 -debug 2
```

아래는 위의 명령에 대한 예시입니다. 실행을 하면 OoVs(Out of Vocabularies, 테스트 데이터에서 보지 못한 단어)와 해당 테스트 문장들에 대한 perplexity가 나오게 됩니다. 주로 문장 수에 대해서 평균을 계산한 (ppl1이 아닌) ppl을 참고 하면 됩니다.

```bash
$ ngram -ppl ./data/test.refined.tok.bpe.txt -lm ./data/ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2
...
...
file ./data/test.refined.tok.bpe.txt: 1000 sentences, 13302 words, 32 OOVs
0 zeroprobs, logprob= -36717.49 ppl= 374.1577 ppl1= 584.7292
```

위의 평가 과정에서는 1,000개의 테스트 문장에 대해서 13,302개의 단어가 포함되어 있었고, 개중에 32개의 OoV가 발생하였습니다. 결국 이 테스트에 대해서는 약 374의 ppl이 측정되었습니다. 이 ppl을 여러가지 하이퍼 파라미터 튜닝 또는 적절한 훈련데이터 추가를 통해서 낮추는 것이 관건이 될 것 입니다.

그리고 -debug 파라미터를 2를 주어, 아래와 같이 좀 더 자세한 각 문장과 단어 별 로그를 볼 수 있습니다. 실제 언어모델 상에서 어떤 n-gram이 일치(hit) 되었는지 이에 대한 확률을 볼 수 있습니다. 3-gram이 없는 경우에는 2-gram이나 1-gram으로 back-off 되는 것을 확인 할 수 있고, back-off 시에는 확률이 굉장히 떨어지는 것을 볼 수 있습니다.

```
▁▁I ▁▁m ▁▁pleased ▁▁with ▁▁the ▁▁way ▁▁we ▁▁handled ▁▁it ▁.
	p( ▁▁I | <s> ) 	= [2gram] 0.06806267 [ -1.167091 ]
	p( ▁▁m | ▁▁I ...) 	= [1gram] 6.597231e-06 [ -5.180638 ]
	p( ▁▁pleased | ▁▁m ...) 	= [1gram] 6.094323e-06 [ -5.215075 ]
	p( ▁▁with | ▁▁pleased ...) 	= [2gram] 0.1292281 [ -0.8886431 ]
	p( ▁▁the | ▁▁with ...) 	= [2gram] 0.05536767 [ -1.256744 ]
	p( ▁▁way | ▁▁the ...) 	= [3gram] 0.003487763 [ -2.457453 ]
	p( ▁▁we | ▁▁way ...) 	= [3gram] 0.1344272 [ -0.8715127 ]
	p( ▁▁handled | ▁▁we ...) 	= [1gram] 1.902798e-06 [ -5.720607 ]
	p( ▁▁it | ▁▁handled ...) 	= [1gram] 0.002173233 [ -2.662894 ]
	p( ▁. | ▁▁it ...) 	= [2gram] 0.05907027 [ -1.228631 ]
	p( </s> | ▁. ...) 	= [3gram] 0.8548805 [ -0.06809461 ]
1 sentences, 10 words, 0 OOVs
0 zeroprobs, logprob= -26.71738 ppl= 268.4436 ppl1= 469.6111
```

위의 결과에서는 10개의 단어가 주어졌고, 5번의 back-off이 되어 3-gram은 3개만 일치되었고, 4개의 2-gram과 4개의 1-gram이 일치되었습니다. 그리하여 -26.71의 로그 확률이 계산되어, 268의 PPL로 환산되었음을 볼 수 있습니다.

### 인터폴레이션

SRILM을 통해서 단순한 스무딩(또는 디스카운팅) 뿐만이 아니라 인터폴레이션을 수행 할 수도 있습니다. 이 경우에는 완성된 두 개의 별도의 언어모델이 필요하고, 이를 섞어주기 위한 하이퍼 파라미터 $\lambda$ (람다)가 필요합니다. 아래와 같이 명령어를 입력하여 인터폴레이션을 수행할 수 있습니다.

```bash
$ ngram -lm <input_lm_fn> -mix-lm <mix_lm_fn> -lambda <mix_ratio_between_0_and_1> -write-lm <output_lm_fn> -debug 2
```

인터폴레이션 이후 성능 평가를 수행하면, 경우에 따라 성능이 향상하는 것을 볼 수 있을 것 입니다. 람다를 튜닝하여 성능 향상의 폭을 더 높일 수 있습니다.
