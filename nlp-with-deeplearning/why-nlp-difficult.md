# Why NLP is difficult?

음성인식은 눈에 보이지 않는 signal을 다룹니다. 보이지도 않는 가운데 noise와 signal을 가려내야 하고, 소리의 특성상 noise와 signal은 그냥 더해져서 나타납니다. 게다가 time-step은 무지하게 길어요. 어려울 것 같습니다. 그렇다면 눈에 보이는 computer vision을 생각 해 보죠. 그런데 computer vision은 눈에 보이지만 이미지는 너무 크고, 다양합니다. 심지어 내 눈에는 다 똑같은 색깔인데 사실은 알고보면 다른 색이라고 합니다. 그럼 애초에 discrete한 단어들로 이루어져 있는 자연어처리를 해 볼까요? 그럼 NLP는 쉬울까요? 하지만 세상에 쉬운일은 없죠... Natural language processing도 다른 분야 못지않게 매우 어렵습니다. 어떠한 점들이 NLP를 어렵게 만드는 것일까요?

## Ambiguity

> 커피숍에 **차**를 마시러 가던 **차** 안에서 나는 그녀에게 **차**였다.
- Google: I was in the car while I was going to drink tea at the coffee shop.
- Microsoft: In a car that was going to drink tea in the coffee shop, I was a car to her.
- Naver: I got dumped by her in the car I was going to the coffee shop.
- Kakao: I was in the car going to the coffee shop for tea and I was tea to her.
- SK: I got dumped by her in the car on the coffee shop.

## Paraphrase

## Discrete, Not Continuous

### Noise

### Curse of Dimensionality