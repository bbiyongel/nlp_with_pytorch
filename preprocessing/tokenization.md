# Sentence Tokenization

## by NLTK
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

## Combine and Tokenization
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

# Part of Speech Tagging, Tokenization (Segmentaion)

우리가 하고자 하는 task에 따라서 Part-of-speech (POS) tagging 또는 단순한 segmentation을 통해 normalization을 수행합니다.

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

## Chinese

### Stanford Parser

### JIEBA