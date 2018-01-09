# Unsupervised NMT

Supervised learning 방식은 높은 정확도를 자랑하지만 labeling 데이터가 필요하기 때문에 데이터 확보, 모델 및 시스템을 구축하는데 높은 비용과 시간이 소요됩니다. 하지만 ***Unsupervised Learning***의 경우에는 데이터 확보에 있어서 훨씬 비용과 시간을 절감할 수 있기 때문에 좋은 대안이 될 수 있습니다.

## Parallel corpus vs Monolingual corpus

그러한 의미에서 parallel corpus에 비해서 확보하기 쉬운 monolingual corpus는 좋은 대안이 될 수 있습니다. 소량의 parallel corpus와 다량의 monolingual corpus를 결합하여 더 나은 성능을 확보할 수도 있을 것입니다. 이전 챕터에 다루었던 [Back translation과 Copied translation](neural-machine-translation/mono.md)에서 이와 관련하여 NMT의 성능을 고도화 하는 방법을 보여주었습니다. 강화학습에서도 마찬가지로 unsupervised 방식을 적용하려는 시도들이 많이 보이고 있습니다. 다만, 대부분의 방식들은 아직 실제 field에서 적용하기에는 다소 효율성이 떨어집니다.

## Dual Learning for Machine Translation

### CycleGAN

먼저 좀 더 이해하기 쉬운 Computer Vision쪽 논문[\[Zhu at el.2017\]](https://arxiv.org/pdf/1703.10593.pdf)을 예제로 설명 해 볼까 합니다. ***Cycle GAN***은 아래와 같이 unparalleled image set이 여러개 있을 때, $$ Set~X $$의 이미지를 $$ Set~Y $$의 이미지로 합성/변환 시켜주는 방법 입니다. 사진을 전체 구조는 유지하되 *모네*의 그림풍으로 바꾸어 주기도 하고, 말과 얼룩말을 서로 바꾸어 주기도 합니다. 겨울 풍경을 여름 풍경으로 바꾸어주기도 합니다.

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)
Cycle GAN - image from [web](https://junyanz.github.io/CycleGAN/)

아래에 이 방법을 도식화 하여 나타냈습니다. $$ Set~X $$와 $$ Set~Y $$ 모두 각각 Generator($$ G, F $$)와 Discriminator($$ D_X, D_Y $$)를 가지고 있어서, $$ minmax $$ 게임을 수행합니다. 

$$ G $$는 $$ x $$를 입력으로 받아 $$ \hat{y} $$으로 변환 해 냅니다. 그리고 $$ D_Y $$는 $$ \hat{y} $$ 또는 $$ y $$를 입력으로 받아 합성 유무($$ Real/Fake $$)를 판단 합니다. 마찬가지로 $$ F $$는 $$ y $$를 입력으로 받아 $$ \hat{x} $$으로 변환 합니다. 이후에 $$ D_X $$는 $$ \hat{x} $$ 또는 $$ x $$를 입력으로 받아 합성 유부를 판단 합니다.

![](/assets/rl-cycle-gan.png)

이 방식의 핵심 key point는 $$ \hat{x} $$나 $$ \hat{y} $$를 합성 할 때에 기존의 Set $$ X, Y $$에 속하는 것 처럼 만들어내야 한다는 것 입니다. 이것을 Machine Translation에 적용 시켜 보면 어떻게 될까요?

### Dual Learning
  
공교롭게도 비슷한 시기에 나온 논문[\[Xia at el.2016\]](https://arxiv.org/pdf/1611.00179.pdf)이 있습니다. ***GAN***이 안되는 NLP의 특성상 CycleGAN처럼 direct로 gradient를 이어줄 수는 없었지만 기본적으로는 아주 비슷한 idea입니다.

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

위와 같이 $$ k $$개의 sampling한 문장에 대해서 각기 방향에 대한 reward를 각각 구한 후, 이를 linear combination을 취해줍니다. 이때, $$ s_{mid} $$는 sampling한 문장을 의미하고, $$ LM_B $$를 사용하여 해당 문장이 $$ language B $$의 집합에 잘 어울리는지를 따져 reward로 리턴합니다. 여기서 기존의 cross entropy를 사용할 수 없는 이유는 monlingual sentence이기 때문에 번역을 하더라도 정답을 알 수 없기 때문입니다.

$$
\triangledown_{\theta_{AB}}\hat{E}[r]=\frac{1}{K}\sum_{k=1}^K{[r_k\triangledown_{\theta_{AB}}\log{P(s_{mid,k}|s;\theta_{AB})}]}
$$
$$
\triangledown_{\theta_{BA}}\hat{E}[r]=\frac{1}{K}\sum_{k=1}^K[(1-\alpha)\triangledown_{\theta_{BA}}\log{P(s|s_{mid,k};\theta_{BA})}]
$$

![](/assets/rl-dual-learning-2.png)

![](/assets/rl-dual-learning-3.png)

## Unsupervised NMT

[\[Artetxe at el.2017\]](https://arxiv.org/pdf/1710.11041.pdf)

![](/assets/rl-unsupervised-nmt-1.png)

![](/assets/rl-unsupervised-nmt-2.png)

  
[\[Lample at el.2017\]](https://arxiv.org/pdf/1711.00043.pdf)



![](/assets/rl-unsupervised-nmt-3.png)

![](/assets/rl-unsupervised-nmt-4.png)

![](/assets/rl-unsupervised-nmt-5.png)

