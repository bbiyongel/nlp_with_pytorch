# Sequence to Sequence

## Architecture Overview
![](/assets/seq2seq_architecture.png)
먼저 번역 또는 seq2seq 모델을 이용한 작업을 간단하게 수식화 해보겠습니다.

$$
\theta^*=argmaxP_\theta(Y|X)~where~X=\{x_1,x_2,\cdots,x_n\},~Y=\{y_1,y_2,\cdots,y_m\}
$$

$$ P(Y|X) $$를 최대로 하는 optimal 모델 파라미터($$ \theta^* $$)를 찾아야 합니다. 즉, source 문장 $$ X $$를 받아서 target 문장 $$ Y $$를 생성 해 내는 작업을 하게 됩니다. 이를 위해서 seq2seq는 크게 3개 서브 모듈로 구성되어 있습니다. 

### 1. Encoder
인코더는 source 문장을 입력으로 받아 문장을 함축하는 의미의 vector로 만들어 냅니다. $$ P(X) $$를 모델링 하는 작업을 수행한다고 볼 수 있습니다. 사실 새로운 형태라기 보단, 이전 챕터에서 다루었던 Text Classificaion에서 사용되었던 RNN 모델과 거의 같다고 볼 수 있습니다. $$ P(X) $$를 모델링하여, 주어진 문장을 vectorize하여 해당 도메인의 hyper-plane에 projection 시키는 작업이라고 할 수 있습니다. -- 이 vector를 classification 하는 것이 text classification 입니다. 다만, 기존의 text classification에서는 모든 정보가 필요하지 않기 때문에 (예를들어 Sentiment Analysis에서는 ***'나는'***과 같이 중립적인 단어는 classification에 필요하지 않기 때문에 정보를 굳이 간직해야 하지 않을 수도 있습니다.) vector로 만들어내는 과정인 정보를 압축함에 있어서 손실 압축을 해도 되는 task이지만, Machine Translation task에 있어서는 이상적으로는 무손실 압축을 해내야 하는 차이는 있습니다.

### 2. Decoder
마찬가지로 디코더도 사실 새로운 형태가 아닙니다. 이전 챕터에서 다루었던 Nerual Network Langauge Model의 연장선으로써, Conditional Neural Network Language Model이라고 할 수 있습니다. 위에서 다루었던 seq2seq모델의 수식을 좀 더 time-step에 대해서 풀어서 써보면 아래와 같습니다.

$$
\theta^*=argmax P_\theta(Y|X)=\prod_{t=1}^{m}P_\theta(y_t|X,y_{t})
$$

### 3. Generator
이 모듈은 Decoder에서 vector를 받아 softmax를 계산하는 단순한 작업을 하는 모듈 입니다. 

## Further use of seq2seq
이와 같이 구성된 Seq2seq 모델은 꼭 기계번역의 task에서만 사용해야 하는 것이 아니라 정말 많은 분야에 적용할 수 있습니다. 특정 도메인의 sequential한 입력을 다른 도메인의 sequential한 데이터로 출력하는데 탁월한 능력을 발휘합니다.

|Seq2seq Applications|Task (From-To)|
|-|-|
|Neural Machine Translation (NMT)|특정 언어 문장을 입력으로 받아 다른 언어의 문장으로 출력|
|Chatbot|사용자의 문장 입력을 받아 대답을 출력|
|Summarization|긴 문장을 입력으로 받아 같은 언어의 요약된 문장으로 출력|
|Other NLP Task|사용자의 문장 입력을 받아 프로그래밍 코드로 출력 등|
|Automatic Speech Recognition (ASR)|사용자의 음성을 입력으로 받아 해당 언어의 문자열(문장)으로 출력|
|Lip Reading|입술 움직임의 동영상을 입력으로 받아 해당 언어의 문장으로 출력|
|Image Captioning|변형된 seq2seq를 사용하여 이미지를 입력으로 받아 그림을 설명하는 문장을 출력|

## Limitation
사실 seq2seq는 AutoEncoder와 굉장히 역할이 비슷하다고 볼 수 있습니다. 그 중에서도 특히 Sequential한 데이터에 대한 task에 강점이 있는 모델이라고 볼 수 있습니다. 하지만 아래와 같은 한계점들이 있습니다.

### 1. Memorization

### 2. Lack of Structural Information

### 3. Domain Expansion

## Code

### 1. Embedding Layer

### 2. Encoder

### 3. Decoder

### 4. Generator


