# Why NLP is difficult?

음성인식은 눈에 보이지 않는 signal을 다룹니다. 보이지도 않는 가운데 noise와 signal을 가려내야 하고, 소리의 특성상 noise와 signal은 그냥 더해져서 나타납니다. 게다가 time-step은 무지하게 길어요. 어려울 것 같습니다. 그렇다면 눈에 보이는 computer vision을 생각 해 보죠. 그런데 computer vision은 눈에 보이지만 이미지는 너무 크고, 다양합니다. 심지어 내 눈에는 다 똑같은 색깔인데 사실은 알고보면 다른 색이라고 합니다. 그럼 애초에 discrete한 단어들로 이루어져 있는 자연어처리를 해 볼까요? 그럼 NLP는 쉬울까요? 하지만 세상에 쉬운일은 없죠... Natural language processing도 다른 분야 못지않게 매우 어렵습니다. 어떠한 점들이 NLP를 어렵게 만드는 것일까요?

사람은 언어를 통해 타인과 교류하고, 의견과 지식을 전달 합니다. 소리로 표현된 말을 석판, 나무, 종이에 적기 시작하였고 사람의 지식은 본격적으로 축적되기 시작하였습니다. 이와 같이 언어는 사람의 생각과 지식을 내포하고 있습니다. 컴퓨터로 하여금 이러한 사람의 언어를 이해할 수 있게 한다면 컴퓨터에게도 지식과 의견을 전달 할 수 있을 것 입니다.

## Ambiguity

아래의 문장을 한번 살펴볼까요. 어떤 회사의 번역이 가장 정확한지 살펴 볼까요. (2018년 3월 기준 입니다.)

> 커피숍에 **차**를 마시러 가던 **차** 안에서 나는 그녀에게 **차**였다.
- Google: I was in the **car** while I was going to drink **tea** at the coffee shop.
- Microsoft: In a **car** that was going to drink **tea** in the coffee shop, I was a **car** to her.
- Naver: I got **dumped** by her in the **car** I was going to the coffee shop.
- Kakao: I was in the **car** going to the coffee shop for **tea** and I was **tea** to her.
- SK: I got **dumped** by her in the **car** on the coffee shop.

안타깝게도 완벽한 번역은 없는 것 같습니다. 같은 **차**라는 단어가 세 번 등장하였고, 모두 다른 의미를 지니고 있습니다: tea, car, and kick (or dump). 일부는 표현을 빠뜨리기도 하였고, 다른 일부는 단어를 헷갈린 것 같습니다. 이렇게 단어의 중의성 때문에 문장을 해석하는데 모호함이 생기기도 합니다. 또 다른 상황을 살펴보겠습니다.

> 나는 철수를 안 때렸다.
1. 철수는 맞았지만, 때린 사람이 나는 아니다.
2. 나는 누군가를 때렸지만, 그게 철수는 아니다.
3. 나는 누군가를 때린 적도 없고, 철수도 맞은 적이 없다.

위와 같이 

> 선생님은 울면서 돌아오는 우리를 위로 했다.
1. (선생님은 울면서) 돌아오는 우리를 위로 했다.
2. 선생님은 (울면서 돌아오는 우리를) 위로 했다.

## Paraphrase

![](http://cdnweb01.wikitree.co.kr/webdata/editor/201608/16/img_20160816082838_215c7a7a.png)
김치 싸대기로 유명한 드라마 [모두 다 김치](https://namu.wiki/w/%EB%AA%A8%EB%91%90%20%EB%8B%A4%20%EA%B9%80%EC%B9%98)의 문제적 장면

영화나 드라마의 어떤 장면을 말로 표현 한다고 해 봅시다. 그럼 아주 다양한 표현이 나올 것 입니다. 하지만 알고보면 다 같은 장면을 묘사하고 있는 것이고, 그 안에 포함된 의미는 같다고 할 수 있을 것 입니다. 이와 같이 문장의 표현 형식은 다양하고, 비슷한 의미의 단어들이 존재하기 때문에 paraphrase의 문제가 존재합니다. 더군다나, 위에서 지적 한 것 처럼 미묘하게 사람들이 이해하고 있는 단어의 의미는 다를 수도 있을 것 입니다. 따라서 이 또한 더욱 paraphrase 문제의 어려움을 가중시킵니다.

## Discrete, Not Continuous

### Noise

### Curse of Dimensionality