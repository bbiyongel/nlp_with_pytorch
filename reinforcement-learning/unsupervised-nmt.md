# Unsupervised NMT

Supervised learning 방식은 높은 정확도를 자랑하지만 labeling 데이터가 필요하기 때문에 데이터 확보, 모델 및 시스템을 구축하는데 높은 비용과 시간이 소요됩니다. 하지만 ***Unsupervised Learning***의 경우에는 데이터 확보에 있어서 훨씬 비용과 시간을 절감할 수 있기 때문에 좋은 대안이 될 수 있습니다.

## Parallel corpus vs Monolingual corpus

그러한 의미에서 parallel corpus에 비해서 확보하기 쉬운 monolingual corpus는 좋은 대안이 될 수 있습니다. 소량의 parallel corpus와 다량의 monolingual corpus를 결합하여 더 나은 성능을 확보할 수도 있을 것입니다. 이전 챕터에 다루었던 [Back translation과 Copied translation](neural-machine-translation/mono.md)에서 이와 관련하여 NMT의 성능을 고도화 하는 방법을 보여주었습니다. 강화학습에서도 마찬가지로 

## Dual Learning for Machine Translation

[\[Zhu at el.2017\]](https://arxiv.org/pdf/1703.10593.pdf)

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

