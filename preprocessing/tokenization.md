# Sentence Tokenization

우리가 다루고자 하는 task들은 입력 단위가 문장단위인 경우가 많습니다. 즉, 한 line에 한 문장만 있어야 합니다. 여러 문장이 한 line에 있거나, 한 문장이 여러 line에 걸쳐 있는 경우에 대해서 sentence tokenization이 필요합니다. 단순이 마침표만을 기준으로 sentence tokenization을 수행하게 되면, U.S. 와 같은 영어 약자나 3.141592와 같은 소숫점 등 여러가지 문제점에 마주칠 수 있습니다. 따라서 직접 tokenization해주기 보다는 NLTK에서 제공하는 tokenizer를 이용하기를 권장합니다. -- 물론, 이 경우에도 완벽하게 처리하지는 못하며, 일부 추가적인 전/후 처리가 필요할 수도 있습니다.

## Tokenization

한 line에 여러 문장이 들어 있는 경우에 대한 Python script 예제.

before:
>현재 TED 웹사이트에는 1,000개가 넘는 TED강연들이 있습니다. 여기 계신 여러분의 대다수는 정말 대단한 일이라고 생각하시겠죠 -- 전 다릅니다. 전 그렇게 생각하지 않아요. 저는 여기 한 가지 문제점이 있다고 생각합니다. 왜냐하면 강연이 1,000개라는 것은, 공유할 만한 아이디어들이 1,000개 이상이라는 뜻이 되기 때문이죠. 도대체 무슨 수로 1,000개나 되는 아이디어를 널리 알릴 건가요?

```python
import sys, fileinput, re
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    for line in fileinput.input():
        if line.strip() != "":
            line = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', line.strip())

            sentences = sent_tokenize(line.strip())

            for s in sentences:
                if s != "":
                    sys.stdout.write(s + "\n")

```

after:
>현재 TED 웹사이트에는 1,000개가 넘는 TED강연들이 있습니다.
여기 계신 여러분의 대다수는 정말 대단한 일이라고 생각하시겠죠 -- 전 다릅니다.
전 그렇게 생각하지 않아요.
저는 여기 한 가지 문제점이 있다고 생각합니다.
왜냐하면 강연이 1,000개라는 것은, 공유할 만한 아이디어들이 1,000개 이상이라는 뜻이 되기 때문이죠.
도대체 무슨 수로 1,000개나 되는 아이디어를 널리 알릴 건가요?

## Combine and Tokenization

여러 line에 한 문장이 들어 있는 경우에 대한 Python script 예제.

before:
>현재 TED 웹사이트에는 1,000개가 넘는 TED강연들이 있습니다.
여기 계신 여러분의 대다수는
정말 대단한 일이라고 생각하시겠죠 --
전 다릅니다. 전 그렇게 생각하지 않아요.
저는 여기 한 가지 문제점이 있다고 생각합니다.
왜냐하면 강연이 1,000개라는 것은,
공유할 만한 아이디어들이 1,000개 이상이라는 뜻이 되기 때문이죠.
도대체 무슨 수로
1,000개나 되는 아이디어를 널리 알릴 건가요?
1,000개의 TED 영상 전부를 보면서

```python
import sys, fileinput
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    buf = []

    for line in fileinput.input():
        if line.strip() != "":
            buf += [line.strip()]
            sentences = sent_tokenize(" ".join(buf))

            if len(sentences) > 1:
                buf = sentences[1:]

                sys.stdout.write(sentences[0] + '\n')

    sys.stdout.write(" ".join(buf) + "\n")
```

after:
>현재 TED 웹사이트에는 1,000개가 넘는 TED강연들이 있습니다.
여기 계신 여러분의 대다수는 정말 대단한 일이라고 생각하시겠죠 -- 전 다릅니다.
전 그렇게 생각하지 않아요.
저는 여기 한 가지 문제점이 있다고 생각합니다.
왜냐하면 강연이 1,000개라는 것은, 공유할 만한 아이디어들이 1,000개 이상이라는 뜻이 되기 때문이죠.
도대체 무슨 수로 1,000개나 되는 아이디어를 널리 알릴 건가요?
1,000개의 TED 영상 전부를 보면서

# Part of Speech Tagging, Tokenization (Segmentaion)

우리가 하고자 하는 task에 따라서 Part-of-speech (POS) tagging 또는 단순한 segmentation을 통해 normalization을 수행합니다. 주로 띄어쓰기에 대해서 살펴 보도록 하겠습니다. 아래에 정리한 프로그램들이 기본적으로 띄어쓰기만 제공하는 프로그램이 아니더라도, tag정보 등을 제외하게 되면 띄어쓰기(segmentation)를 수행하는 것과 동일합니다.

띄어쓰기(tokenization or segmentation)에 대해서 살펴 보겠습니다. 한국어, 영어와 달리 일본어, 중국어의 경우에는 띄어쓰기가 없는 경우가 없습니다. 또한, 한국어의 경우에도 띄어쓰기가 도입된 것은 근대에 이르러서이기 때문에, 띄어쓰기의 표준화가 부족하여 띄어쓰기가 중구난방인 경우가 많습니다. 특히나, 한국어의 경우에는 띄어쓰기가 문장의 의미 해석에 큰 영향을 끼치지 않기 때문에 더욱이 이런 현상은 가중화 됩니다. 따라서, 한국어의 경우에는 띄어쓰기가 이미 되어 있지만, 제각각인 경우가 많기 때문에 normalization을 해주는 의미로써 다시한번 적용 됩니다. 또한, 교착어로써 접사를 어근에서 분리해주는 역할도 하여, sparseness를 해소하는 역할도 합니다. 이러한 한국어의 띄어쓰기 특성은 앞서 [Why Korean NLP is Hell](nlp-with-deeplearning/korean-is-hell.md)에서 다루었습니다.

일본어, 중국어의 경우에는 띄어쓰기가 없기 때문에, 애초에 모든 문장들이 같은 띄어쓰기 형태(띄어쓰기가 없는 형태)를 띄고 있지만, 적절한 language model 구성을 위하여 띄어쓰기가 필요합니다. 다만, 중국어의 경우에는 character 단위로 문장을 실제 처리하더라도 크게 문제가 없기도 합니다.

영어의 경우에는 기본적으로 띄어쓰기가 있고, 대부분의 경우 매우 잘 표준을 따르고 있습니다. 다만 language model을 구성함에 있어서 좀 더 용이하게 해 주기 위한 일부 처리들을 해주면 더 좋습니다. NLTK를 사용하여 이러한 전처리를 수행합니다.

각 언어별 주요 프로그램들을 정리하면 아래와 같습니다.

|언어|프로그램명|제작언어|특징|
|-|-|-|-|
|한국어|Mecab|C++|일본어 Mecab을 wrapping하였으며, 속도가 가장 빠름. 설치가 종종 까다로움|
|한국어|KoNLPy|Python Wrapping(복합)|PIP을 통해 설치 가능하며 사용이 쉬우나, 일부 tagger의 경우에는 속도가 느림|
|일본어|Mecab|C++|속도가 빠르다|
|중국어|Stanford Parser|Java|미국 스탠포드에서 개발|
|중국어|PKU Parser|Java|북경대에서 개발. Stanford Parser와 성능 차이가 거의 없음|
|중국어|Jieba|Python|가장 최근에 개발됨. Python으로 제작되어 시스템 구성에 용이|

사실 일반적인 또는 전형적인 쉬운 문장(표준어를 사용하며 문장 구조가 명확한 문장)의 경우에는 대부분의 프로그램들의 성능이 비슷합니다. 하지만 각 프로그램 성능의 차이를 만드는 것은 바로 신조어나 보지 못한 고유명사를 처리하는 능력입니다. 따라서 어떤 한 프로그램을 선택하고자 할때에, 그런 부분에 초점을 맞추어 성능을 평가하고 선택해야 할 필요성이 있습니다.

## Korean

### Mecab

한국어 tokenization에 가장 많이 사용되는 프로그램은 Mecab입니다. 원래 Mecab은 일본어 tokenizer 오픈소스로 개발되었지만, 이를 한국어 tokenizing에 성공적으로 적용시켜 널리 사용되고 있습니다. 아래와 같이 설치 가능합니다. -- 참고사이트: http://konlpy-ko.readthedocs.io/ko/v0.4.3/install/#ubuntu

```bash
$ sudo apt-get install curl
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

위의 instruction을 따라 정상적으로 설치 된 Mecab의 경우에는, KoNLPy에서 불러 사용할 수 있습니다. -- 참고사이트: http://konlpy-ko.readthedocs.io/ko/v0.4.3/api/konlpy.tag/#mecab-class

또는 아래와 같이 bash상에서 **mecab**명령어를 통해서 직접 실행 가능하며 standard input/output을 사용하여 tokenization을 수행할 수 있습니다.

```bash
$ echo "안녕하세요, 반갑습니다!" | mecab
안녕	NNG,*,T,안녕,*,*,*,*
하	XSV,*,F,하,*,*,*,*
세요	EP+EF,*,F,세요,Inflect,EP,EF,시/EP/*+어요/EF/*
,	SC,*,*,*,*,*,*,*
반갑	VA,*,T,반갑,*,*,*,*
습니다	EF,*,F,습니다,*,*,*,*
!	SF,*,*,*,*,*,*,*
EOS
```

만약, 단순히 띄어쓰기만 적용하고 싶을 경우에는 **-O wakati** 옵션을 사용하여 실행시키면 됩니다. 마찬가지로 standard input/output을 통해 수행 할 수 있습니다.

```bash
$ echo "안녕하세요, 반갑습니다!" | mecab -O wakati
안녕 하 세요 , 반갑 습니다 !
```

위와 같이 기본적으로 어근과 접사를 분리하는 역할을 수행하며, 또는 잘못된 띄어쓰기에 대해서도 올바르게 교정을 해 주는 역할 또한 수행 합니다. 어근과 접사를 분리함으로써 어휘의 숫자를 효과적으로 줄여 sparsity를 해소할 수 있습니다.

### KoNLPy

KoNLPy는 여러 한국어 tokenizer 또는 tagger들을 모아놓은 wrapper를 제공합니다. 이름에서 알 수 있듯이 Python wrapper를 제공하므로 시스템 연동 및 구성이 용이할 수 있습니다. 하지만 내부 라이브러리들은 각기 다른 언어(Java, C++ 등)로 이루어져 있어 호환성에 문제가 생기기도 합니다. 또한, 일부 library들은 상대적으로 제작/업데이트가 오래되었고, java로 구현되어 있어 C++로 구현되어 있는 Mecab에 비해 속도면에서 불리함을 갖고 있습니다. 따라서 대용량의 corpus를 처리할 때에 불리하기도 합니다. 하지만 설치 및 사용이 쉽고 다양한 library들을 모아놓았다는 장점 때문에 사랑받고 널리 이용되고 있습니다.

> 참고사이트: http://konlpy-ko.readthedocs.io/

## English 

### Tokenizer NLTK (Moses)

영어의 경우에는 위에서 언급했듯이, 기본적인 띄어쓰기가 잘 통일되어 있는 편이기 때문에, 띄어쓰기 자체에 대해서는 큰 normalization 이슈가 없습니다. 다만 comma, period, quotation 등을 띄어주어야 합니다. NLTK를 통해서 이런 처리를 수행하게 됩니다. 아래는 NLTK를 사용한 Python script 예제 입니다.

> 참고사이트: http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.moses

before:
>North Korea's state mouthpiece, the Rodong Sinmun, is also keeping mum on Kim's summit with Trump while denouncing ever-tougher U.S. sanctions on the rogue state.

```python
import sys, fileinput
from nltk.tokenize.moses import MosesTokenizer

t = MosesTokenizer()

if __name__ == "__main__":
    for line in fileinput.input():
        if line.strip() != "":
            tokens = t.tokenize(line.strip(), escape=False)

            sys.stdout.write(" ".join(tokens) + "\n")
```

after:
>North Korea 's state mouthpiece , the Rodong Sinmun , is also keeping mum on Kim 's summit with Trump while denouncing ever-tougher U.S. sanctions on the rogue state .

## Chinese

### Stanford Parser

> 참고사이트: https://nlp.stanford.edu/software/lex-parser.shtml

### JIEBA

비교적 가장 늦게 개발/업데이트가 이루어진 Jieba library는, 사실 성능상에서 다른 parser들과 큰 차이가 없지만, python으로 구현되어 있어 우리가 실제 상용화를 위한 배치 시스템을 구현 할 때에 매우 쉽고 용이할 수 있습니다. 특히, 딥러닝을 사용한 머신러닝 시스템은 대부분 Python위에서 동작하기 때문에 일관성 있는 시스템을 구성하기 매우 좋습니다.

> 참고사이트: https://github.com/fxsjy/jieba