# Exploit Duality

## What is Duality?

Duality란 무엇일까요? 우리가 보통 기계학습을 통해 학습하는 것은 어떤 도메인의 데이터 $$X$$를 받아서, 다른 도메인의 데이터 $$Y$$로 맵핑(mapping)해주는 함수를 근사(approximation)하는 것이라 할 수 있습니다. 따라서 대부분의 기계학습에 사용되는 데이터셋은 두 도메인 사이의 데이터로 구성되어있기 마련입니다.

| Task($$ D_1 \rightarrow D_2$$) | Domain 1 | Domain 2 | Task($$ D_1 \leftarrow D_2$$) |
| --- | --- | --- | --- |
| 기계번역 | source 언어 문장 | target 언어 문장 | 기계번역 |
| 음성인식 | 음성 신호 | 텍스트(transcript) | 음성합성 |
| 이미지 분류 | 이미지 | class | 이미지 합성 |
| 요약 | 본문(content) 텍스트 | 제목(title) 텍스트 | 본문 생성 |

위와 같이 두 도메인 사이의 데이터의 관계를 배우는 방향에 따라서 음성인식이 되기도 하고, 음성합성이 되기도 합니다. 이러한 두 도메인 사이의 관계를 duality라고 우리는 정의 합니다. 대부분의 기계학습 문제들은 이와 같이 duality를 가지고 있는데, 특히 기계번역은 각 도메인의 데이터 간에 정보량의 차이가 거의 없는 것이 가장 큰 특징이자 장점 입니다. 따라서 duality를 가장 적극적으로 활용할 수 있습니다.

## CycleGAN

먼저 좀 더 이해하기 쉬운 duality의 활용 예로, 컴퓨터 비전(Computer Vision)쪽 논문[\[Zhu at el.2017\]](https://arxiv.org/pdf/1703.10593.pdf)을 설명 해 볼까 합니다. Cycle GAN은 아래와 같이 unparalleled image set이 여러개 있을 때, $$ Set~X $$의 이미지를 $$ Set~Y $$의 이미지로 합성/변환 시켜주는 방법 입니다. 사진을 전체 구조는 유지하되 모네의 그림풍으로 바꾸어 주기도 하고, 말과 얼룩말을 서로 바꾸어 주기도 합니다. 겨울 풍경을 여름 풍경으로 바꾸어주기도 합니다.

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)
Cycle GAN - image from [web](https://junyanz.github.io/CycleGAN/)

아래에 이 방법을 도식화 하여 나타냈습니다. $$ Set~X $$와 $$ Set~Y $$ 모두 각각 Generator($$ G, F $$)와 Discriminator($$ D_X, D_Y $$)를 가지고 있어서, min/max 게임을 수행합니다. 

$$ G $$는 $$ x $$를 입력으로 받아 $$ \hat{y} $$으로 변환 해 냅니다. 그리고 $$ D_Y $$는 $$ \hat{y} $$ 또는 $$ y $$를 입력으로 받아 합성 유무($$ Real/Fake $$)를 판단 합니다. 마찬가지로 $$ F $$는 $$ y $$를 입력으로 받아 $$ \hat{x} $$으로 변환 합니다. 이후에 $$ D_X $$는 $$ \hat{x} $$ 또는 $$ x $$를 입력으로 받아 합성 유부를 판단 합니다.

![](/assets/rl-cycle-gan.png)

이 방식의 핵심 key point는 $$ \hat{x} $$나 $$ \hat{y} $$를 합성 할 때에 기존의 Set $$ X, Y $$에 속하는 것 처럼 만들어내야 한다는 것 입니다. 이것을 Machine Translation에 적용 시켜 보면 어떻게 될까요?
