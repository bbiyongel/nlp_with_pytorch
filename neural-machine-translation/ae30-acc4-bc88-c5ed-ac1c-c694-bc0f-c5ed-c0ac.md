# Machine Translation \(MT\)

기계 번역(Machine Translation)은 단순히 언어를 번역하는 것이 아닌, 자연언어처리 영역에서의 종합예술이라 할 수 있습니다. Neural Machine Translation (NMT)는 end-to-end learning 으로써, Rule Based Machine Translation (RBMT), Statistical Machine Translation (SMT)로 이어져온 기계 번역의 30년 역사 중에서 가장 큰 성취를 이루어냈습니다. 이번 챕터는 NMT에 대한 인사이트를 얻을 수 있도록, seq2seq with attention의 동작 방식/원리를 이해하고, 더 나아가 응용 방법에 대해서도 소개 합니다. 또한, 기계번역 시스템을 만들기 위한 프로세스와 최신 연구 동향을 아울러 소개 합니다.

## History

### 1. Rule based Machine Translation

### 2. Statistical Machine Translation

### 3. Neural Machine Translation

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/basic_nmt_architecture.png)

사실, 딥러닝 이전의 AI의 전성기(1980년대)에도 Neural Network을 사용하여 Machine Translation 문제를 해결하려는 시도는 여럿 있었습니다. 하지만 당시에도 $$ Encoder \longrightarrow Decoder $$ 형태의 구조를 가지고 있었지만, 당연히 지금과 같은 성능을 내기는 어려웠습니다.

### a. Invasion of NMT

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/progress_in_machine_translation.png)

### b. Why it works well?