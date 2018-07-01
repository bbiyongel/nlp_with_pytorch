# Exploit Duality

## What is Duality?

| Task($$ D_1 \rightarrow D_2$$) | Domain 1 | Domain 2 | Task($$ D_1 \leftarrow D_2$$) |
| --- | --- | --- | --- |
| 기계번역 | source 언어 문장 | target 언어 문장 | 기계번역 |
| 음성인식 | 음성 신호 | 텍스트(transcript) | 음성합성 |
| 이미지 분류 | 이미지 | class | 이미지 합성 |
| 요약 | 본문(content) 텍스트 | 제목(title) 텍스트 | 본문 생성 |

## CycleGAN

먼저 좀 더 이해하기 쉬운 Computer Vision쪽 논문[\[Zhu at el.2017\]](https://arxiv.org/pdf/1703.10593.pdf)을 예제로 설명 해 볼까 합니다. ***Cycle GAN***은 아래와 같이 unparalleled image set이 여러개 있을 때, $$ Set~X $$의 이미지를 $$ Set~Y $$의 이미지로 합성/변환 시켜주는 방법 입니다. 사진을 전체 구조는 유지하되 *모네*의 그림풍으로 바꾸어 주기도 하고, 말과 얼룩말을 서로 바꾸어 주기도 합니다. 겨울 풍경을 여름 풍경으로 바꾸어주기도 합니다.

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)
Cycle GAN - image from [web](https://junyanz.github.io/CycleGAN/)

아래에 이 방법을 도식화 하여 나타냈습니다. $$ Set~X $$와 $$ Set~Y $$ 모두 각각 Generator($$ G, F $$)와 Discriminator($$ D_X, D_Y $$)를 가지고 있어서, $$ minmax $$ 게임을 수행합니다. 

$$ G $$는 $$ x $$를 입력으로 받아 $$ \hat{y} $$으로 변환 해 냅니다. 그리고 $$ D_Y $$는 $$ \hat{y} $$ 또는 $$ y $$를 입력으로 받아 합성 유무($$ Real/Fake $$)를 판단 합니다. 마찬가지로 $$ F $$는 $$ y $$를 입력으로 받아 $$ \hat{x} $$으로 변환 합니다. 이후에 $$ D_X $$는 $$ \hat{x} $$ 또는 $$ x $$를 입력으로 받아 합성 유부를 판단 합니다.

![](/assets/rl-cycle-gan.png)

이 방식의 핵심 key point는 $$ \hat{x} $$나 $$ \hat{y} $$를 합성 할 때에 기존의 Set $$ X, Y $$에 속하는 것 처럼 만들어내야 한다는 것 입니다. 이것을 Machine Translation에 적용 시켜 보면 어떻게 될까요?
