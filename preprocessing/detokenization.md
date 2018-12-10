# 분절 복원

아래는 전처리 과정에서 분절을 수행 하고, 다시 복원(detokenization)하는 과정을 설명 한 것입니다. 하나씩 따라가보도록 하겠습니다.

아래와 같은 영어 원문이 주어지면,
```
There's currently over a thousand TED Talks on the TED website.
```
각 언어 별 분절기 모듈(영어의 경우 NLTK)을 통해 분절을 수행하고, 기존의 띄어쓰기와 이번에 분절기에 의해 수행된 공백과의 구분을 위해 특수문자 '▁'를 원래의 공백 위치에 삽입합니다. 다만 이 ▁ 문자는 기존의 _와 다른 문자로써 특수기호 입니다.
```
▁There 's ▁currently ▁over ▁a ▁thousand ▁TED ▁Talks ▁on ▁the ▁TED ▁website .
```
여기에 서브워드 단위 분절을 수행하며 이전 과정까지의 공백과 서브워드 단위 분절로 인한 공백을 구분하기 위한 특수문자 '▁'를 삽입합니다.
```
▁▁There ▁'s ▁▁currently ▁▁over ▁▁a ▁▁thous and ▁▁TED ▁▁T al ks ▁▁on ▁▁the ▁▁TED ▁▁we b site ▁.
```
이렇게 전처리 과정이 종료 되었습니다. 이런 형태의 문절 된 문장을 자연어처리 모델에 훈련시키면 똑같은 형태로 분절 된 문장을 생성 해 낼 것 입니다. 그럼 이런 문장을 복원(detokenization)하여 사람이 읽기 좋은 형태로 만들어 주어야 합니다.

먼저 공백(whitespace)을 제거합니다.
```
▁▁There▁'s▁▁currently▁▁over▁▁a▁▁thousand▁▁TED▁▁Talks▁▁on▁▁the▁▁TED▁▁website▁.
```
그리고 특수문자 '▁'가 2개가 동시에 있는 문자열 '▁▁'을 공백으로 치환합니다. 그럼 한 개 짜리 '▁'만 남습니다.
```
There▁'s currently over a thousand TED Talks on the TED website▁.
```
마지막 남은 특수문자 '▁'를 제거합니다. 그럼 문장 복원이 완성 됩니다.
```
There's currently over a thousand TED Talks on the TED website.
```

## 분절 후처리

위의 예제에서처럼 분절에 대한 복원을 수월하게 하기 위하여, 분절 이후에는 특수문자를 알맞은 자리에 삽입해주어야 합니다. 아래는 기존의 공백과 전처리 한 단계로 인해 생성된 공백을 구분하기 위한 특수문자 '▁'을 삽입하는 파이썬 스크립트 에제 입니다.

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

위 스크립트의 사용 방법은 아래와 같습니다. 주로 다른 분절 모듈의 수행 후에 바로 붙여 사용하여 좋습니다.

```bash
$ cat [before filename] | python tokenizer.py | python post_tokenize.py [before filename]
```

## 분절 복원 예제

아래는 앞서 설명한 복원(detokenization) bash에서 sed 명령어를 통해 수행 할 경우에 대한 예제 입니다.

```bash
sed "s/ //g" | sed "s/▁▁/ /g" | sed "s/▁//g" | sed "s/^\s//g"
```

또는 아래의 파이썬 스크립트 예제 처럼 처리 할 수도 있습니다.

```python
import sys

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            line = line.strip().replace(' ', '').replace('▁▁', ' ').replace('▁', '').strip()

            sys.stdout.write(line + '\n')
```