# Machine Translation \(MT\)

기계 번역(Machine Translation)은 단순히 언어를 번역하는 것이 아닌, 자연언어처리 영역에서의 종합예술이라 할 수 있습니다. 사실 이번 챕터를 위해서 이전 챕터들을 다루었다고 해도 과언이 아닙니다. Neural Machine Translation (NMT)는 end-to-end learning 으로써, Rule Based Machine Translation (RBMT), Statistical Machine Translation (SMT)로 이어져온 기계 번역의 30년 역사 중에서 가장 큰 성취를 이루어냈습니다. 이번 챕터는 NMT에 대한 인사이트를 얻을 수 있도록, seq2seq with attention의 동작 방식/원리를 이해하고, 더 나아가 응용 방법에 대해서도 소개 합니다. 또한, 기계번역 시스템을 만들기 위한 프로세스와 최신 연구 동향을 아울러 소개 합니다.

## History

### 1. Rule based Machine Translation

전통적인 방식의 번역이라고 할 수 있습니다. 우리가 흔히 어릴때 부터 배워 온 방식입니다. 문장의 구조를 분석하고, 그 분석에 따라 규칙을 세우고 분류를 나누어서 정해진 규칙에 따라서 번역을 합니다. 밑에서 다룰 SMT에 비해서 자연스러운 표현이 가능하지만, 그 규칙을 일일히 사람이 만들어내야 하므로 번역기를 만드는데 많은 자원과 시간이 소모됩니다. 따라서 번역 언어쌍을 확장하는데에 있어서도 굉장히 불리합니다.

### 2. Statistical Machine Translation

NMT이전에 세상을 지배하던 번역 방식입니다. Google이 자신의 번역 시스템에 도입하면서 더욱 유명해졌습니다. 이 시스템 또한 여러가지 모듈로 이루어져 있고, 굉장히 복잡합니다. 통계기반 방식을 사용하므로 언어쌍을 확장하는데 있어서 기존의 RBMT에 비하여 유리하였습니다.

### 3. Neural Machine Translation

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/basic_nmt_architecture.png)

사실, 딥러닝 이전의 AI의 전성기(1980년대)에도 Neural Network을 사용하여 Machine Translation 문제를 해결하려는 시도는 여럿 있었습니다. 하지만 당시에도 $$ Encoder \longrightarrow Decoder $$ 형태의 구조를 가지고 있었지만, 당연히 지금과 같은 성능을 내기는 어려웠습니다.

### a. Invasion of NMT

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/progress_in_machine_translation.png)

현재의 NMT 방식이 제안되고, 곧 기존의 SMT 방식을 크게 앞질러 버렸습니다. 이제는 구글의 번역기에도 NMT 방식이 사용됩니다.

### b. Why it works well?

왜 Neural Machine Translation이 잘 동작하는 것일까요?

#### 1. 
#### 2.
#### 3.
#### 4.