# Sequence to Sequence

## Architecture Overview

![](/assets/seq2seq_architecture.png)

먼저 번역 또는 seq2seq 모델을 이용한 작업을 간단하게 수식화 해보겠습니다.


$$
\theta^*=argmaxP_\theta(Y|X)~where~X=\{x_1,x_2,\cdots,x_n\},~Y=\{y_1,y_2,\cdots,y_m\}
$$


$$ P(Y|X) $$를 최대로 하는 optimal 모델 파라미터\($$ \theta^* $$\)를 찾아야 합니다. 즉, source 문장 $$ X $$를 받아서 target 문장 $$ Y $$를 생성 해 내는 작업을 하게 됩니다. 이를 위해서 seq2seq는 크게 3개 서브 모듈로 구성되어 있습니다.

### Encoder

인코더는 source 문장을 입력으로 받아 문장을 함축하는 의미의 vector로 만들어 냅니다. $$ P(X) $$를 모델링 하는 작업을 수행한다고 볼 수 있습니다. 사실 새로운 형태라기 보단, 이전 챕터에서 다루었던 Text Classificaion에서 사용되었던 RNN 모델과 거의 같다고 볼 수 있습니다. $$ P(X) $$를 모델링하여, 주어진 문장을 vectorize하여 해당 도메인의 hyper-plane에 projection 시키는 작업이라고 할 수 있습니다. -- 이 vector를 classification 하는 것이 text classification 입니다.

![](/assets/encoder_sentence_projection.png)

다만, 기존의 text classification에서는 모든 정보가 필요하지 않기 때문에 \(예를들어 Sentiment Analysis에서는 _**'나는'**_과 같이 중립적인 단어는 classification에 필요하지 않기 때문에 정보를 굳이 간직해야 하지 않을 수도 있습니다.\) vector로 만들어내는 과정인 정보를 압축함에 있어서 손실 압축을 해도 되는 task이지만, Machine Translation task에 있어서는 이상적으로는 무손실 압축을 해내야 하는 차이는 있습니다.

$$
h_{t}^{src} = RNN_{enc}(emb_{src}(x_t), h_{t-1}^{src})
$$
$$
H^{src} = [h_{1}^{src}; h_{2}^{src}; \cdots; h_{n}^{src}]
$$

Encoder를 수식으로 나타내면 위와 같습니다. $$[;]$$는 concatenate를 의미합니다. 위의 수식은 time-step별로 GRU를 통과시킨 것을 나타낸 것이고, 사실상 실제 코딩을 하게 되면 아래와 같이 됩니다.

$$
H^{src} = RNN_{enc}(emb_{src}(X), h_{0}^{src})
$$

### Decoder

마찬가지로 디코더도 사실 새로운 형태가 아닙니다. 이전 챕터에서 다루었던 Nerual Network Langauge Model의 연장선으로써, Conditional Neural Network Language Model이라고 할 수 있습니다. 위에서 다루었던 seq2seq모델의 수식을 좀 더 time-step에 대해서 풀어서 써보면 아래와 같습니다.

$$
P_\theta(Y|X)=\prod_{t=1}^{m}P_\theta(y_t|X,y_{<t})
$$
$$
\log P_\theta(Y|X) = \sum_{t=1}^{m}\log P_\theta(y_t|X, y_{<t})
$$

보면 RNNLM의 수식에서 조건부에 $$ X $$가 추가 된 것을 확인 할 수 있습니다. 즉, 이전 time-step의 단어들과 주어진 encoder의 정보에 기반해서 현재 time-step의 단어를 유추해 내는 작업을 수행합니다.

$$
h_{t}^{tgt} = RNN_{dec}(emb_{tgt}(y_{t-1}), h_{t-1}^{tgt})~~where~h_{0}^{tgt} = h_{n}^{src} and ~y_{0}=BOS
$$

위의 수식은 decoder를 나타낸 것입니다. 특기할 점은 decoder 입력의 초기값으로써, $$ y_0 $$에 ***BOS***를 넣어준다는 것 입니다. 

### Generator

이 모듈은 아래와 같이 Decoder에서 vector를 받아 softmax를 계산하는 단순한 작업을 하는 모듈 입니다. $$ |Y|=m $$일때, $$ y_{m} $$은 ***EOS*** 토큰이 됩니다. 주의할 점은 이 마지막 $$ y_{m} $$은 decoder 계산의 종료를 나타내기 때문에, decoder의 입력으로 들어가는 일이 없습니다.

$$
\hat{y}_{t}=softmax(linear_{hs \rightarrow |V_{tgt}|}(h_{t}^{tgt}))~~and~\hat{y}_{m}=EOS
$$
$$
where~hs~is~hidden~size~of~RNN,~and~|V_{tgt}|~is~size~of~output~vocabulary.
$$

## Teacher Forcing

많은 분들이 여기까지 잘 따라왔다면 궁금즘을 하나 가질 수 있습니다. Decoder의 입력으로 이전 time-step의 출력이 들어가는것이 훈련 때도 같은 것인가?

Decoder를 구현할 때에 중요한 점은, **training 시에는 decoder의 입력으로 이전 time-step의 decoder의 출력값이 아닌, 실제 $$ Y $$가 들어간다**는 것입니다. 하지만, inference 할 때에는 실제 $$ Y $$를 모르기 때문에, 이전 time-step에서 계산되어 나온 $$ \hat{y_{t-1}} $$를 decoder의 입력으로 사용합니다. 이 방법을 ***Teacher Forcing***이라고 합니다. -- 추후 만악의 근원(?)이 됩니다.

Teacher Forcing이 필요한 이유는 NMT의 수식을 살펴보면 알 수 있습니다. 해당 time-step의 단어를 구할 때 수식은 아래와 같습니다.

$$
\hat{y}_t=argmax{P(y_t|y_{<t},X)}~where~X=\{x_1,x_2,\cdots,x_n\}
$$

위와 같이 조건부에 $$ \hat{y}_{<t} $$가 들어가는 것이 아닌, $$ y_{<t} $$가 들어가는 것이기 때문에, 훈련시에 이전 time-step의 출력을 넣어줄 수 없습니다. 만약 넣어주게 된다면 해당 time-step의 decoder에겐 잘못된 것을 가르쳐 주는 꼴이 될 것입니다.

따라서 training 할 때에는 모든 time-step을 한번에 계산할 수 있습니다. 그러므로 decoder도 각 time-step별이 아닌 한번에 수식을 정리할 수 있습니다.

$$
H^{tgt}=RNN_{dec}(emb_{tgt}([BOS;Y[:-1]]),h_{n}^{src})
$$

## Applications of seq2seq

이와 같이 구성된 Seq2seq 모델은 꼭 기계번역의 task에서만 사용해야 하는 것이 아니라 정말 많은 분야에 적용할 수 있습니다. 특정 도메인의 sequential한 입력을 다른 도메인의 sequential한 데이터로 출력하는데 탁월한 능력을 발휘합니다.

| Seq2seq Applications | Task \(From-To\) |
| --- | --- |
| Neural Machine Translation \(NMT\) | 특정 언어 문장을 입력으로 받아 다른 언어의 문장으로 출력 |
| Chatbot | 사용자의 문장 입력을 받아 대답을 출력 |
| Summarization | 긴 문장을 입력으로 받아 같은 언어의 요약된 문장으로 출력 |
| Other NLP Task | 사용자의 문장 입력을 받아 프로그래밍 코드로 출력 등 |
| Automatic Speech Recognition \(ASR\) | 사용자의 음성을 입력으로 받아 해당 언어의 문자열\(문장\)으로 출력 |
| Lip Reading | 입술 움직임의 동영상을 입력으로 받아 해당 언어의 문장으로 출력 |
| Image Captioning | 변형된 seq2seq를 사용하여 이미지를 입력으로 받아 그림을 설명하는 문장을 출력 |

## Autoencoder?

## Limitation

사실 seq2seq는 [AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder)와 굉장히 역할이 비슷하다고 볼 수 있습니다. 그 중에서도 특히 Sequential한 데이터에 대한 task에 강점이 있는 모델이라고 볼 수 있습니다. 하지만 아래와 같은 한계점들이 있습니다.

### Memorization

Neural Network 모델은 데이터를 압축하는데에 탁월한 성능\([Manifold Assumption 참고](https://en.wikipedia.org/wiki/Semi-supervised_learning#Manifold_assumption)\)을 지녔습니다.  하지만, Neural Network은 [도라에몽의 주머니](https://namu.wiki/w/4차원 주머니#toc)처럼 무한하게 정보를 집어넣을 수 없습니다. 따라서 표현할 수 있는 정보는 한계가 있기 때문에, 문장\(또는 sequence\)가 길어질수록 기억력이 떨어지게 됩니다. 비록 LSTM이나 GRU를 사용함으로써 성능을 끌어올릴 수 있지만, 한계가 있기 마련입니다.

### Lack of Structural Information

현재 주류의 Deeplearning NLP는 문장을 이해함에 있어서 구조 정보를 사용하기보단, 단순히 sequential한 데이터로써 다루는 경향이 있습니다. 비록 이러한 접근방법은 현재까지 대성공을 거두고 있지만, 다음 단계로 나아가기 위해서는 구조 정보도 필요할 것이라 생각하는 사람들이 많습니다.

### Chatbot?

사실 이 항목은 단점이라기보다는 그냥 당연한 이야기일 수 있습니다. seq2seq는 sequential한 데이터를 입력으로 받아서 다른 도메인의 sequential한 데이터로 출력하는 능력이 뛰어납니다. 따라서, 처음에는 많은 사람들이 seq2seq를 잘 훈련시키면 Chatbot의 기능도 어느정도 할 수 있지 않을까 하는 기대를 했습니다. 하지만 자세히 생각해보면, 대화의 흐름에서 _**대답**_은 _**질문**_에 비해서 새로운 정보\(지식-knowledge, 문맥-context\)가 추가 된 경우가 많습니다. 따라서 기존의 typical한 seq2seq의 task\(번역, 요약\)등은 새로운 정보의 추가가 없기 때문에 잘 해결할 수 있었지만, 대화의 경우에는 좀 더 발전된 architecture가 필요할 것 입니다.

## Code

### Encoder

```python
class Encoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers = 4, dropout_p = .2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(word_vec_dim, int(hidden_size / 2), num_layers = n_layers, dropout = dropout_p, bidirectional = True, batch_first = True)

    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_dim)

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first = True)
        else:
            x = emb
        
        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first = True)

        return y, h
```

### Decoder

```python
class Decoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers = 4, dropout_p = .2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(word_vec_dim + hidden_size, hidden_size, num_layers = n_layers, dropout = dropout_p, bidirectional = False, batch_first = True)

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (batch_size, 1, word_vec_dim)
        # |h_t_1_tilde| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([emb_t, h_t_1_tilde], dim = -1)
        y, h = self.rnn(x, h_t_1)

        return y, h
```

### Generator

```python
class Generator(nn.Module):
    
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)

        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)

        return y
```

### Sequence-to-Sequence (Combine)

### Loss

seq2seq는 기본적으로 classification task이므로, $$ Cross Entropy $$을 _**Loss Function**_으로 사용합니다. 다만, GPU memory 문제와 Padding의 이슈가 있으므로 유의해야 할 점들이 있습니다.

