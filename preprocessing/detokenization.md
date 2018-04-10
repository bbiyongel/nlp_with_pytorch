# Detokenization

>There's currently over a thousand TED Talks on the TED website.

>▁There 's ▁currently ▁over ▁a ▁thousand ▁TED ▁Talks ▁on ▁the ▁TED ▁website .

>▁▁There ▁'s ▁▁currently ▁▁over ▁▁a ▁▁thous and ▁▁TED ▁▁T al ks ▁▁on ▁▁the ▁▁TED ▁▁we b site ▁.

>▁▁There▁'s▁▁currently▁▁over▁▁a▁▁thousand▁▁TED▁▁Talks▁▁on▁▁the▁▁TED▁▁website▁.

> There▁'s currently over a thousand TED Talks on the TED website▁.

> There's currently over a thousand TED Talks on the TED website.

## Post Tokenization

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

## Detokenization

```bash
sed "s/ //g" | sed "s/▁▁/ /g" | sed "s/▁//g"
```

```python
import sys

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            line = line.strip().replace(' ', '').replace('▁▁', ' ').replace('▁', '')

            sys.stdout.write(line + '\n')
```