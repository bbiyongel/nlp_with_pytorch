# n-gram Exercise with SRILM

SRILM은 음성인식, segmentation, 기계번역 등에 사용되는 통계 언어 모델 (n-gram language model)을 구축하고 적용 할 수 있는 toolkit입니다. 이 책에서 다루는 다른 알고리즘이나 기법들에 비하면, SRI speech research lab에서 1995년부터 연구/개발 해 온 유서깊은(?) toolkit 입니다.

## Install SRILM

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

위와 같이 매번 sed와 regular expression을 통하는 것이 번거롭다면, preprocessing 챕터에서 구현한 detokenization.py python script를 통하여 detokenization을 수행 할 수도 있습니다.

### Evaluation

이렇게 language model을 훈련하고 나면 test set에 대해서 evaluation을 통해 얼마나 훌륭한 langauge model이 만들어졌는지 체크 할 필요가 있습니다. Language model에 대한 성능평가는 아래와 같은 명령을 통해 수행 될 수 있습니다.

```bash
$ ngram -ppl <test_fn> -lm <input_lm_fn> -order 3 -debug 2
```

아래는 위의 명령에 대한 예시입니다. 실행을 하면 OOVs(Out of Vocabularies)와 해당 test 문장들에 대한 perplexity가 나오게 됩니다. 주로 문장 수에 대해서 normalize를 수행한 (ppl1이 아닌) ppl을 참고 하면 됩니다.

```bash
$ ngram -ppl ./data/test.refined.tok.bpe.txt -lm ./data/ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2
...
...
file ./data/test.refined.tok.bpe.txt: 1000 sentences, 13302 words, 32 OOVs
0 zeroprobs, logprob= -36717.49 ppl= 374.1577 ppl1= 584.7292
```

위의 evaluation에서는 1,000개의 테스트 문장에 대해서 13,302개의 단어가 포함되어 있었고, 개중에 32개의 OOV가 발생하였습니다. 결국 이 테스트에 대해서는 약 374의 ppl이 측정되었습니다. 이 ppl을 여러가지 hyper-parameter 튜닝 또는 적절한 훈련데이터 추가를 통해서 낮추는 것이 관건이 될 것 입니다.

그리고 -debug 파라미터를 2를 주게 되면 아래와 같이 좀 더 자세한 각 문장과 단어 별 log를 볼 수 있습니다. 실제 language model 상에서 어떤 n-gram이 hit되었는지와 이에 대한 확률을 볼 수 있습니다. 3-gram이 없는 경우에는 2-gram이나 1-gram으로 back-off 되는 것을 확인 할 수 있고, back-off 시에는 확률이 굉장히 떨어지는 것을 볼 수 있습니다. 따라서 back-off를 통해서 unseen word sequence에 대해서 generalization을 할 수 있었지만, 여전히 성능에는 아쉬움이 남는 것을 알 수 있습니다.

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

위의 결과에서는 10개의 단어가 주어졌고, 5번의 back-off이 되어 3-gram은 3개만 hit되었고, 4개의 2-gram과 4개의 1-gram이 hit되었습니다. 그리하여 -26.71의 log-probability가 계산되어, 268의 PPL로 환산되었음을 볼 수 있습니다.

### Interpolation

SRILM을 통해서 단순한 smoothing(or discounting) 뿐만이 아니라 interpolation을 수행 할 수도 있습니다. 이 경우에는 완성된 두 개의 별도의 language model이 필요하고, 이를 섞어주기 위한 hyper parameter lambda가 필요합니다. 아래와 같이 명령어를 입력하여 interpolation을 수행할 수 있습니다.

```bash
$ ngram -lm <input_lm_fn> -mix-lm <mix_lm_fn> -lambda <mix_ratio_between_0_and_1> -write-lm <output_lm_fn> -debug 2
```