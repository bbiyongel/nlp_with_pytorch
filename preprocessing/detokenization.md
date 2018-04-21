# Detokenization

아래는 preprocessing에서 tokenization을 수행 하고, 다시 복원(detokenization)하는 과정을 설명 한 것입니다. 하나씩 따라가보도록 하겠습니다.

아래와 같은 영어 원문이 주어지면,
```
There's currently over a thousand TED Talks on the TED website.
```
각 언어 별 tokenizer(영어의 경우 NLTK)를 통해 tokenization을 수행하고, 기존의 띄어쓰기와 tokenization에 의해 수행된 공백과의 구분을 위해 **▁**을 원래의 공백 위치에 삽입합니다.
```
▁There 's ▁currently ▁over ▁a ▁thousand ▁TED ▁Talks ▁on ▁the ▁TED ▁website .
```
여기에 subword segmentation을 수행하며 이전 step까지의 공백과 subword segmentation으로 인한 공백을 구분하기 위한 **▁**를 삽입합니다.
```
▁▁There ▁'s ▁▁currently ▁▁over ▁▁a ▁▁thous and ▁▁TED ▁▁T al ks ▁▁on ▁▁the ▁▁TED ▁▁we b site ▁.
```
이렇게 preprocessing 과정이 종료되었습니다. 이런 형태의 tokenized문장을 NLP 모델에 훈련시키면 같은 형태로 tokenized된 문장을 생성 해 낼 것 입니다. 그럼 이런 문장을 복원(detokenization)하여 사람이 읽기 좋은 형태로 만들어 주어야 합니다.

먼저 whitespace를 제거합니다.
```
▁▁There▁'s▁▁currently▁▁over▁▁a▁▁thousand▁▁TED▁▁Talks▁▁on▁▁the▁▁TED▁▁website▁.
```
그리고 **▁**가 2개가 동시에 있는 문자열 **▁▁**을 white space로 치환합니다. 그럼 한 개 짜리 **▁**만 남습니다.
```
There▁'s currently over a thousand TED Talks on the TED website▁.
```
마지막 남은 **▁**를 제거합니다. 그럼 문장 복원이 완성 됩니다.
```
There's currently over a thousand TED Talks on the TED website.
```

## Post Tokenization

아래는 기존의 공백과 전처리 한 단계로 인해 생성된 공백을 구분하기 위한 **▁**을 삽입하는 python script 에제 입니다.

```python
import sys

STR = '▁'

if __name__ == "__main__":
    ref_fn = sys.argv[1]

    f = open(ref_fn, 'r')

    for ref in f:
        ref_tokens = ref.strip().split(' ')
        tokens = sys.stdin.readline().strip().split(' ')

        idx = 0
        buf = []

        # We assume that stdin has more tokens than reference input.
        for ref_token in ref_tokens:
            tmp_buf = []

            while idx < len(tokens):
                tmp_buf += [tokens[idx]]
                idx += 1

                if ''.join(tmp_buf) == ref_token:
                    break

            if len(tmp_buf) > 0:
                buf += [STR + tmp_buf[0]] + tmp_buf[1:]

        sys.stdout.write(' '.join(buf) + '\n')

    f.close()
```

위 script의 사용 방법은 아래와 같습니다. 주로 다른 tokenizer 수행 후에 바로 붙여 사용하여 좋습니다.

```bash
$ cat [before filename] | python tokenizer.py | python post_tokenize.py [before filename]
```

## Detokenization

아래는 앞서 설명한 detokenization을 bash에서 sed 명령어를 통해 수행 할 경우에 대한 예제 입니다. 다만, 이 경우에는 최종 문장의 맨 앞에 공백이 생깁니다.

```bash
sed "s/ //g" | sed "s/▁▁/ /g" | sed "s/▁//g"
```

또는 아래의 python script 예제 처럼 처리 할 수도 있습니다.

```python
import sys

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            line = line.strip().replace(' ', '').replace('▁▁', ' ').replace('▁', '').strip()

            sys.stdout.write(line + '\n')
```