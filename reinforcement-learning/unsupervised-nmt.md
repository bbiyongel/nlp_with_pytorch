# Unsupervised NMT

Supervised learning 방식은 높은 정확도를 자랑하지만 labeling 데이터가 필요하기 때문에 데이터 확보, 모델 및 시스템을 구축하는데 높은 비용과 시간이 소요됩니다. 하지만 ***Unsupervised Learning***의 경우에는 데이터 확보에 있어서 훨씬 비용과 시간을 절감할 수 있기 때문에 좋은 대안이 될 수 있습니다.

## Parallel corpus vs Monolingual corpus

그러한 의미에서 parallel corpus에 비해서 확보하기 쉬운 monolingual corpus는 좋은 대안이 될 수 있습니다. 소량의 parallel corpus와 다량의 monolingual corpus를 결합하여 더 나은 성능을 확보할 수도 있을 것입니다. 이전 챕터에 다루었던 [Back translation과 Copied translation](neural-machine-translation/mono.md)에서 이와 관련하여 NMT의 성능을 고도화 하는 방법을 보여주었습니다. 강화학습에서도 마찬가지로 unsupervised 방식을 적용하려는 시도들이 많이 보이고 있습니다. 다만, 대부분의 방식들은 아직 실제 field에서 적용하기에는 다소 효율성이 떨어집니다.

## Dual Learning for Machine Translation

먼저 좀 더 이해하기 쉬운 Computer Vision쪽 논문[\[Zhu at el.2017\]](https://arxiv.org/pdf/1703.10593.pdf)을 예제로 설명 해 볼까 합니다. ***Cycle GAN***은 아래와 같이 unparalleled image set이 여러개 있을 때, $$ Set~X $$의 이미지를 $$ Set~Y $$의 이미지로 합성/변환 시켜주는 방법 입니다. 사진을 전체 구조는 유지하되 *모네*의 그림풍으로 바꾸어 주기도 하고, 말과 얼룩말을 서로 바꾸어 주기도 합니다. 겨울 풍경을 여름 풍경으로 바꾸어주기도 합니다.

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)
Cycle GAN - image from [web](https://junyanz.github.io/CycleGAN/)

아래에 이 방법을 도식화 하여 나타냈습니다. $$ Set~X $$와 $$ Set~Y $$ 모두 각각 Generator($$ G, F $$)와 Discriminator($$ D_X, D_Y $$)를 가지고 있어서, $$ minmax $$ 게임을 수행합니다. 

$$ G $$는 $$ x $$를 입력으로 받아 $$ \hat{y} $$으로 변환 해 냅니다. 그리고 $$ D_Y $$는 $$ \hat{y} $$ 또는 $$ y $$를 입력으로 받아 합성 유무($$ Real/Fake $$)를 판단 합니다. 
![](/assets/rl-cycle-gan.png)

  
[\[Xia at el.2016\]](https://arxiv.org/pdf/1611.00179.pdf)



![](/assets/rl-dual-learning-1.png)

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

