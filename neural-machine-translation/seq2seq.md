# Sequence to Sequence

## Architecture Overview
![](/assets/seq2seq_architecture.png)
먼저 번역 또는 seq2seq 모델을 이용한 작업을 간단하게 수식화 해보겠습니다.

$$
\theta^*=argmaxP_\theta(Y|X)~where~X=\{x_1,x_2,\cdots,x_n\},~Y=\{y_1,y_2,\cdots,y_m\}
$$

$$ P(Y|X) $$를 최대로 하는 optimal 모델 파라미터($$ \theta^* $$)를 찾아야 합니다. 즉, source 문장 $$ X $$를 받아서 target 문장 $$ Y $$를 생성 해 내는 작업을 하게 됩니다. 이를 위해서 seq2seq는 크게 3개 서브 모듈로 구성되어 있습니다. 

### 1. Encoder
인코더는 source 문장을 입력으로 받아 문장을 함축하는 의미의 vector로 만들어 냅니다. $$ P(X) $$를 모델링 하는 작업을 수행한다고 볼 수 있습니다.

### 2. Decoder

### 3. Generator
이 모듈은 Decoder에서 vector를 받아 softmax를 계산하는 단순한 작업을 하는 모듈 입니다. 

## Further use of seq2seq
이와 같이 구성된 Seq2seq 모델은 꼭 기계번역의 task에서만 사용해야 하는 것이 아니라 정말 많은 분야에 적용할 수 있습니다. 특정 도메인의 sequential한 입력을 다른 도메인의 sequential한 데이터로 출력하는데 탁월한 능력을 발휘합니다.

|Task|Description|
|-|-|
|Neural Machine Translation (NMT)||
|Chatbot||
|Other NLP Task||
|Automatic Speech Recognition (ASR)||
|Image Captioning||
|Lip Reading||
|||
|||

## Limitation

## Code

### 1. Embedding Layer

### 2. Encoder

### 3. Decoder

### 4. Generator


