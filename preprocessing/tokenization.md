# Sentence Tokenization

한국어, 영어와 달리 일본어, 중국어의 경우에는 띄어쓰기가 없는 경우가 없습니다. 또한, 한국어의 경우에도 띄어쓰기가 도입된 것은 근대에 이르러서이기 때문에, 띄어쓰기의 표준화가 부족하여 띄어쓰기가 중구난방인 경우가 많습니다. 특히나, 한국어의 경우에는 띄어쓰기가 문장의 의미 해석에 큰 영향을 끼치지 않기 때문에 더욱이 이런 현상은 가중화 됩니다. 이러한 한국어의 띄어쓰기 특성은 앞서 [Why Korean NLP is Hell](nlp-with-deeplearning/korean-is-hell.md)에서 다루었습니다.

## Tokenization

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

>현재 TED 웹사이트에는 1,000개가 넘는 TED강연들이 있습니다.
여기 계신 여러분의 대다수는 정말 대단한 일이라고 생각하시겠죠 -- 전 다릅니다.
전 그렇게 생각하지 않아요.
저는 여기 한 가지 문제점이 있다고 생각합니다.
왜냐하면 강연이 1,000개라는 것은, 공유할 만한 아이디어들이 1,000개 이상이라는 뜻이 되기 때문이죠.
도대체 무슨 수로 1,000개나 되는 아이디어를 널리 알릴 건가요?

## Combine and Tokenization

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

>현재 TED 웹사이트에는 1,000개가 넘는 TED강연들이 있습니다.
여기 계신 여러분의 대다수는 정말 대단한 일이라고 생각하시겠죠 -- 전 다릅니다.
전 그렇게 생각하지 않아요.
저는 여기 한 가지 문제점이 있다고 생각합니다.
왜냐하면 강연이 1,000개라는 것은, 공유할 만한 아이디어들이 1,000개 이상이라는 뜻이 되기 때문이죠.
도대체 무슨 수로 1,000개나 되는 아이디어를 널리 알릴 건가요?
1,000개의 TED 영상 전부를 보면서

# Part of Speech Tagging, Tokenization (Segmentaion)

우리가 하고자 하는 task에 따라서 Part-of-speech (POS) tagging 또는 단순한 segmentation을 통해 normalization을 수행합니다.

띄어쓰기(tokenization or segmentation)에 대해서 살펴 보겠습니다. 한국어, 영어와 달리 일본어, 중국어의 경우에는 띄어쓰기가 없는 경우가 없습니다. 또한, 한국어의 경우에도 띄어쓰기가 도입된 것은 근대에 이르러서이기 때문에, 띄어쓰기의 표준화가 부족하여 띄어쓰기가 중구난방인 경우가 많습니다. 특히나, 한국어의 경우에는 띄어쓰기가 문장의 의미 해석에 큰 영향을 끼치지 않기 때문에 더욱이 이런 현상은 가중화 됩니다. 이러한 한국어의 띄어쓰기 특성은 앞서 [Why Korean NLP is Hell](nlp-with-deeplearning/korean-is-hell.md)에서 다루었습니다.

각 언어별 주요 tokenizer를 정리하면 아래와 같습니다.

|언어|프로그램명|특징|주소|
|-|-|-|-|
|한국어||||
|한국어||||
|일본어||||
|중국어||||
|중국어||||
|중국어||||

## Korean

### Mecab

```bash
$ sudo apt-get install curl
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
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
```bash
$ echo "안녕하세요, 반갑습니다!" | mecab -O wakati
안녕 하 세요 , 반갑 습니다 !
```

### KoNLPy

http://konlpy-ko.readthedocs.io/

## English 

### Tokenizer NLTK (Moses)

http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.moses

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

>North Korea 's state mouthpiece , the Rodong Sinmun , is also keeping mum on Kim 's summit with Trump while denouncing ever-tougher U.S. sanctions on the rogue state .

## Chinese

### Stanford Parser

https://nlp.stanford.edu/software/lex-parser.shtml

### JIEBA

https://github.com/fxsjy/jieba