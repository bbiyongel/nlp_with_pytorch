# Unsupervised NMT

Supervised learning 방식은 높은 정확도를 자랑하지만 labeling 데이터가 필요하기 때문에 데이터 확보, 모델 및 시스템을 구축하는데 높은 비용과 시간이 소요됩니다. 하지만 ***Unsupervised Learning***의 경우에는 데이터 확보에 있어서 훨씬 비용과 시간을 절감할 수 있기 때문에 좋은 대안이 될 수 있습니다.

## 1. Parallel corpus vs Monolingual corpus

그러한 의미에서 parallel corpus에 비해서 확보하기 쉬운 monolingual corpus는 좋은 대안이 될 수 있습니다. 소량의 parallel corpus와 다량의 monolingual corpus를 결합하여 더 나은 성능을 확보할 수도 있을 것입니다. 이전 챕터에 다루었던 [Back translation과 Copied translation](neural-machine-translation/mono.md)에서 이와 관련하여 NMT의 성능을 고도화 하는 방법을 보여주었습니다. 강화학습에서도 마찬가지로 unsupervised 방식을 적용하려는 시도들이 많이 보이고 있습니다. 다만, 대부분의 방식들은 아직 실제 field에서 적용하기에는 다소 효율성이 떨어집니다.

## 2. Dual Learning for Machine Translation

### a. CycleGAN

먼저 좀 더 이해하기 쉬운 Computer Vision쪽 논문[\[Zhu at el.2017\]](https://arxiv.org/pdf/1703.10593.pdf)을 예제로 설명 해 볼까 합니다. ***Cycle GAN***은 아래와 같이 unparalleled image set이 여러개 있을 때, $$ Set~X $$의 이미지를 $$ Set~Y $$의 이미지로 합성/변환 시켜주는 방법 입니다. 사진을 전체 구조는 유지하되 *모네*의 그림풍으로 바꾸어 주기도 하고, 말과 얼룩말을 서로 바꾸어 주기도 합니다. 겨울 풍경을 여름 풍경으로 바꾸어주기도 합니다.

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)
Cycle GAN - image from [web](https://junyanz.github.io/CycleGAN/)

아래에 이 방법을 도식화 하여 나타냈습니다. $$ Set~X $$와 $$ Set~Y $$ 모두 각각 Generator($$ G, F $$)와 Discriminator($$ D_X, D_Y $$)를 가지고 있어서, $$ minmax $$ 게임을 수행합니다. 

$$ G $$는 $$ x $$를 입력으로 받아 $$ \hat{y} $$으로 변환 해 냅니다. 그리고 $$ D_Y $$는 $$ \hat{y} $$ 또는 $$ y $$를 입력으로 받아 합성 유무($$ Real/Fake $$)를 판단 합니다. 마찬가지로 $$ F $$는 $$ y $$를 입력으로 받아 $$ \hat{x} $$으로 변환 합니다. 이후에 $$ D_X $$는 $$ \hat{x} $$ 또는 $$ x $$를 입력으로 받아 합성 유부를 판단 합니다.

![](/assets/rl-cycle-gan.png)

이 방식의 핵심 key point는 $$ \hat{x} $$나 $$ \hat{y} $$를 합성 할 때에 기존의 Set $$ X, Y $$에 속하는 것 처럼 만들어내야 한다는 것 입니다. 이것을 Machine Translation에 적용 시켜 보면 어떻게 될까요?

### b. Dual Learning
  
공교롭게도 비슷한 시기에 나온 논문[\[Xia at el.2016\]](https://arxiv.org/pdf/1611.00179.pdf)이 있습니다. ***GAN***이 안되는 NLP의 특성상 CycleGAN처럼 direct로 gradient를 이어줄 수는 없었지만 기본적으로는 아주 비슷한 idea입니다. 짝이 없는 monolingual corpus를 이용하여 성능을 극대화 하고자 하였습니다.

즉, monolingual sentence($$ s $$)에 대해서 번역을 하고 그 문장($$ s_{mid} $$)을 사용하여 복원을 하였을 때($$ \hat{s} $$) 원래의 처음 문장으로 돌아올 수 있도록(처음 문장과의 차이를 최소화 하도록) 훈련하는 것입니다. 이때, 번역된 문장 $$ s_{mid} $$는 자연스러운 해당 언어의 문장이 되었는가도 중요한 지표가 됩니다.

![](/assets/rl-dual-learning-1.png)

위에서 설명한 algorithm을 따라가 보겠습니다. 이 방법에서는 $$ Set~X,~Set~Y $$ 대신에 $$ Language~A,~Language~B $$로 표기하고 있습니다. $$ G_{A \rightarrow B} $$의 파라미터 $$ \theta_{AB} $$와 $$ F_{B \rightarrow A} $$의 파라미터 $$ \theta_{BA} $$가 등장합니다. 이 $$ G_{A \rightarrow B}, F_{B \rightarrow A} $$는 모두 parallel corpus에 의해서 pre-training이 되어 있는 상태 입니다. 즉, 기본적인 저성능의 번역기 수준이라고 가정합니다.

우리는 기존의 policy gradient와 마찬가지로 아래와 같은 parameter update를 수행해야 합니다.

$$
\theta_{AB} \leftarrow \theta_{AB} + \gamma \triangledown_{\theta_{AB}}\hat{E}[r]
$$
$$
\theta_{BA} \leftarrow \theta_{BA} + \gamma \triangledown_{\theta_{BA}}\hat{E}[r]
$$

$$ \hat{E}[r] $$을 각각의 parameter에 대해서 미분 해 준 값을 더해주는 것을 볼 수 있습니다. 이 reward의 기대값은 아래와 같이 구할 수 있습니다.

$$
r=\alpha r_{AB} + (1-\alpha)r_{BA}
$$
$$
r_{AB}=LM_{B}(s_{mid})
$$
$$
r_{BA}=\log{P(s|s_{mid};\theta_{BA})}
$$

위와 같이 $$ k $$개의 sampling한 문장에 대해서 각기 방향에 대한 reward를 각각 구한 후, 이를 linear combination을 취해줍니다. 이때, $$ s_{mid} $$는 sampling한 문장을 의미하고, $$ LM_B $$를 사용하여 해당 문장이 $$ language B $$의 집합에 잘 어울리는지를 따져 reward로 리턴합니다. 여기서 기존의 cross entropy를 사용할 수 없는 이유는 monlingual sentence이기 때문에 번역을 하더라도 정답을 알 수 없기 때문입니다. 또한 우리는 다수의 monolingual corpus를 갖고 있기 때문에, $$ LM $$은 쉽게 만들어낼 수 있습니다.

$$
\triangledown_{\theta_{AB}}\hat{E}[r]=\frac{1}{K}\sum_{k=1}^K{[r_k\triangledown_{\theta_{AB}}\log{P(s_{mid,k}|s;\theta_{AB})}]}
$$
$$
\triangledown_{\theta_{BA}}\hat{E}[r]=\frac{1}{K}\sum_{k=1}^K[(1-\alpha)\triangledown_{\theta_{BA}}\log{P(s|s_{mid,k};\theta_{BA})}]
$$

이렇게 얻어진 $$ E[r] $$를 각 parameter에 대해서 미분하게 되면 위와 같은 수식을 얻을 수 있고, 상기 서술한 parameter update 수식에 대입하면 됩니다. 비슷한 방식으로 $$ B \rightarrow A $$를 구할 수 있습니다.

![](/assets/rl-dual-learning-2.png)

위의 table은 이 방법의 성능을 비교한 결과 입니다. Pseudo-NMT는 이전 챕터에서 설명하였던 back-translation을 의미합니다. 그리고 그 방식보다 더 좋은 성능을 기록한 것을 볼 수 있습니다.

![](/assets/rl-dual-learning-3.png)

또한, 위 그래프에서 문장의 길이와 상관 없이 모든 구간에서 baseline NMT를 성능으로 압도하고 있는 것을 알 수 있습니다. 다만, parallel corpus의 양이 커질수록 monolingual corpus에 의한 성능 향상의 폭이 줄어드는 것을 확인 할 수 있습니다.

## 3. Unsupervised NMT

위의 Dual Learning 논문과 달리 이 논문[\[Lample at el.2017\]](https://arxiv.org/pdf/1711.00043.pdf)은 오직 Monolingual Corpus만을 사용하여 번역기를 제작하는 방법을 제안하였습니다. 따라서 **Unsupervised NMT**라고 할 수 있습니다.

이 논문의 핵심 아이디어는 아래와 같습니다. 제일 중요한 것은 encoder가 언어에 상관 없이 같은 내용일 경우에 같은 vector로 encoding할 수 있도록 훈련하는 것 입니다. 이러한 encoder를 만들기 위해서 GAN이 도입되었습니다.

GAN을 NLP에 쓰지 못한다고 해 놓고 GAN을 썼다니 이게 무슨 소리인가 싶겠지만, encoder의 출력값인 vector에 대해서 GAN을 적용한 것이라 discrete한 값이 아닌 continuous한 값이기 때문에 가능한 것 입니다.

![](/assets/rl-unsupervised-nmt-3.png)

이렇게 다른 언어일지라도 동일한 내용에 대해서는 같은 vector로 encoding하도록 훈련 된 encoder의 출력값을 가지고 decoder로 원래의 문장으로 잘 돌아오도록 해 주는 것이 이 논문의 핵심 내용입니다.

특기 할 만한 점은 이 논문에서는 언어에 따라서 encoder와 decoder를 다르게 사용한 것이 아니라 언어에 상관없이 1개씩의 encoder와 decoder를 사용하였습니다.

이 논문의 훈련은 3가지 관점에서 수행됩니다.

### a. Denoising Autoencoder

이전 챕터에서 다루었듯이 Seq2seq 모델도 결국 Autoencoder의 일종이라고 볼 수 있습니다. 그러한 관점에서 autoencoder(AE)로써 단순 복사(copy) task는 굉장히 쉬운 task에 속합니다. 그러므로 단순히 encoding 한 source sentence를 같은 언어의 문장으로 decoding 하는 것은 매우 쉬운 일이 될 것입니다. 따라서 AE에게 단순히 복사 작업을 지시하는 것이 아닌 noise를 섞어 준 source sentence에서 denoising을 하면서 reconstruction(복원)을 할 수 있도록 훈련해야 합니다. 따라서 이 task의 objective는 아래와 같습니다.

$$
\mathcal{L}_{auto}(\theta_{enc},\theta_{dec},\mathcal{Z},\ell)=\Bbb{E}_{x\sim\mathcal{D}_\ell,\hat{x}\sim d(e(C(x),\ell),\ell)}[\triangle(\hat{x},x)]
$$

$$ \hat{x}\sim d(e(C(x),\ell),\ell) $$는 source sentence $$ x $$를 $$ C $$를 통해 noise를 추가하고, 같은 언어 $$ \ell $$로 encoding과 decoding을 수행한 것을 의미합니다. $$ \triangle(\hat{x},x) $$는 원문과 복원된 문장과의 차이(error)를 나타냅니다.

#### Noise Model

***Noise Model*** $$ C(x) $$는 임의로 문장 내 단어들을 drop하거나, 순서를 섞어주는 일을 합니다. drop rate는 보통 0.1, 순서를 섞어주는 단어사이의 거리는 3정도가 적당한 것으로 설명 합니다.

### b. Cross Domain Training (Translation)

이번엔 이전 iteration의 모델 $$ M $$에서 언어($$ \ell_2 $$)의 noisy translated된 문장($$ y $$)을 다시 언어($$ \ell_1 $$) source sentence로 원상복구 하는 task에 대한 objective 입니다.

$$
y=M(x)
$$
$$
\mathcal{L}_{cd}(\theta_{enc},\theta_{dec},\mathcal{Z},\ell_1,\ell_2)=\Bbb{E}_{x\sim\mathcal{D}_{\ell_1},\hat{x}\sim d(e(C(y),\ell_2),\ell_1)}[\triangle(\hat{x},x)]
$$

### c. Adversarial Training

Encoder가 언어와 상관없이 항상 같은 분포로 hyper plane에 projection하는지 검사하기 위한 ***discriminator***가 추가되어 Adversarial Training을 진행합니다. 

Discriminator는 latent variable $$ z $$의 언어를 예측하여 아래의 cross-entropy loss를 minimize하도록 훈련됩니다. $$ x_i, \ell_i $$는 같은 언어(language pair)를 의미합니다.

$$
\mathcal{L}_D(\theta_D|\theta,\mathcal{Z})=-\Bbb{E}_{(x_i,\ell_i)}[\log{p_D(\ell_i|e(x_i,\ell_i))}]
$$

따라서 encoder는 discriminator를 속일 수 있도록(***fool***) 훈련 되야 합니다.

$$
$$


![](/assets/rl-unsupervised-nmt-4.png)

![](/assets/rl-unsupervised-nmt-5.png)

