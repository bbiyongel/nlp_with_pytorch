# 왜 자연어처리는 어려운가

음성인식은 눈에 보이지 않는 신호를 다룹니다. 보이지도 않는 가운데 노이즈(noise)와 신호를 가려내야 하고, 소리의 특성상 노이즈와 신호는 그냥 더해져서 나타납니다. 게다가 time-step은 매우 많습니다. 어려울 것 같습니다. 그렇다면 눈에 보이는 영상처리 분야를 생각 해 보죠. 그런데 이미지 데이터는 눈에 보이지만 이미지는 너무 크고, 다양합니다. 심지어 내 눈에는 다 똑같은 색깔인데 사실은 알고보면 다른 색이라고 합니다. 그럼 애초에 discrete한 단어들로 이루어져 있는 자연어처리를 해 볼까요? 그럼 자연어처리는 쉬울까요? 하지만 세상에 쉬운 일은 없습니다. 자연어처리도 다른 분야 못지않게 매우 어렵습니다. 어떠한 점들이 자연어처리를 어렵게 만드는 것일까요?

사람은 언어를 통해 타인과 교류하고, 의견과 지식을 전달 합니다. 소리로 표현된 말을 석판, 나무, 종이에 적기 시작하였고 사람의 지식은 본격적으로 축적되기 시작 하였습니다. 이와 같이 언어는 사람의 생각과 지식을 내포하고 있습니다. 컴퓨터로 하여금 이러한 사람의 언어를 이해할 수 있게 한다면 컴퓨터에게도 지식과 의견을 전달 할 수 있을 것 입니다.

## 모호성

아래의 문장을 한번 살펴볼까요. 어떤 회사의 번역이 가장 정확한지 살펴 볼까요. (2018년 4월 기준)

|원문|차를 마시러 공원에 가던 차 안에서 나는 그녀에게 차였다.|
|-|-|
|Google|I was kicking her in the car that went to the park for tea.|
|Microsoft|I was a car to her, in the car I had a car and went to the park.|
|Naver|I got dumped by her on the way to the park for tea.|
|Kakao|I was in the car going to the park for tea and I was in her car.|
|SK|I got dumped by her in the car that was going to the park for a cup of tea.|

안타깝게도 완벽한 번역은 없는 것 같습니다. 같은 '차'라는 단어가 세 번 등장하였고, 모두 다른 의미를 지니고 있습니다: tea, car, and kick (or dump). 일부는 표현을 빠뜨리기도 하였고, 다른 일부는 단어를 헷갈린 것 같습니다. 이렇게 단어의 중의성 때문에 문장을 해석하는데 모호함이 생기기도 합니다. 또 다른 상황을 살펴보겠습니다.

|원문|나는 철수를 안 때렸다.|
|-|-|
|해석#1|철수는 맞았지만, 때린 사람이 나는 아니다.|
|해석#2|나는 누군가를 때렸지만, 그게 철수는 아니다.|
|해석#3|나는 누군가를 때린 적도 없고, 철수도 맞은 적이 없다.|

위와 같이 문장 내 정보의 부족으로 인한 모호성이 발생 할 수 있습니다. 사람의 언어는 마치 생명체와 같아서 계속해서 진화합니다. 이때 언어는 그 효율성을 극대화 하도록 발전하였기 때문에, 최소한의 표현으로 최대한의 정보를 표현하려 합니다. 따라서 최대한 쉽고 뻔한 정보는 생략하고 문장으로 표현하곤 합니다. 사람은 이렇게 생략된 정보로 생긴 구멍을 쉽게 메울 수 있지만, 컴퓨터의 경우에는 이러한 문제가 매우 어렵게 다가옵니다. 사실 첫 예제의 '차'도 주변 단어들의 문맥(context)을 보면 중의성을 해소(word sense disambiguation)할 수 있습니다. 아래의 예제도 문장 내 정보의 부족이 야기한 구조 해석의 문제입니다.

|원문|선생님은 울면서 돌아오는 우리를 위로 했다.|
|-|-|
|해석#1|(선생님은 울면서) 돌아오는 우리를 위로 했다.|
|해석#2|선생님은 (울면서 돌아오는 우리를) 위로 했다.|

이러한 단어 의미의 모호성으로 인해 생기는 문제는 추후 단어 의미에 대한 챕터에서 다루도록 합니다.

## 다양한 표현

|번호|표현|
|-|-|
|1.|골든 리트리버 한마리가 잔디밭에서 공중의 원반을 향해 달려가고 있습니다.|
|2.|원반이 날라가는 방향으로 개가 뛰어가고 있습니다.|
|3.|개가 잔디밭에서 원반을 쫒아가고 있습니다.|
|4.|잔디밭에서 강아지가 프리스비를 향해 뛰어가고 있습니다.|
|5.|높이 던져진 원반을 향해 멍멍이가 신나게 뛰어갑니다.|
|6.|노란 개가 원반을 잡으러 뛰어가고 있습니다.|

예를 들어 잔디반에서 개가 공중에 던져진 원반을 잡기 위헤 달려가고 있는 사진이 있다고 해보겠습니다. 이 사진을 사람들에게 한 문장으로 묘사해달라고 한다면, 다양한 표현이 나올 것 입니다. 하지만 알고보면 다 같은 사진을 묘사하고 있는 것이고, 그 안에 포함된 의미는 같다고 할 수 있을 것 입니다. 이와 같이 문장의 표현 형식은 다양하고, 비슷한 의미의 단어들이 존재하기 때문에 다양한 표현의 문제가 존재합니다. 더군다나, 위에서 지적 한 것 처럼 미묘하게 사람들이 이해하고 있는 단어의 의미는 다를 수도 있을 것 입니다. 따라서 이 또한 더욱 이 문제의 어려움을 가중시킵니다.

## 불연속적(Discrete, Not Continuous) 데이터

사실은 discrete하기 때문에 그동안 쉽다고 느껴졌습니다. 하지만 딥러닝(인공신경망)에 적용하기 위해서는 continuous한 값으로 바꾸어주어야 합니다. 단어 임베딩이 그 역할을 훌륭하게 수행하고 있습니다. 하지만 애초에 continuous한 값이 아니었기 때문에 딥러닝에서 여러가지 방법을 구현할 때에 제약이 존재합니다.

### 차원의 저주

Discrete한 데이터이기 때문에 많은 종류의 데이터를 표현하기 위해서는 데이터의 종류 만큼의 엄청난 차원이 필요합니다. 즉, 각 단어를 discrete한 심볼로 다루었기 때문에, 마치 어휘의 크기 $=|V|$ 만큼의 차원이 있는 것이나 마찬가지였습니다. 이러한 희소성(sparseness) 해결하기 위해서 단어를 적절하게 분절(segmentation)하는 등 여러가지 노력이 필요하였습니다. 다행히 적절한 단어 임베딩을 통해서 차원 축소(dimension reduction)를 하여 이 문제를 해결함으로써, 이제는 이러한 문제는 예전보다 크게 다가오진 않습니다. 

이에 대해서는 추후 단어 임베딩 챕터에서 좀 더 자세히 다루도록 하겠습니다.

### 노이즈와 정규화

모든 분야의 데이터에서 노이즈를 신호로 부터 적절히 분리해 내는 일은 매우 중요합니다. 그러한 과정에서 자칫 실수하면 데이터는 본래의 의미마저 같이 잃어버릴 수도 있습니다. 이러한 관점에서 자연어처리는 어려움이 존재 합니다. 특히, 다른 종류의 데이터에 비해서 데이터가 살짝 바뀌었을 때의 의미의 변화가 훨씬 크기 때문 입니다. 예를 들어 이미지에서 한 픽셀의 RGB값이 각각 0에서 255까지로 나타내어지고, 그 값중 하나의 수치가 1이 바뀌었다고 해도 해당 이미지의 의미는 변화가 없다고 할 수 있습니다. 하지만 단어는 discrete한 심볼이기 때문에, 단어가 살짝만 바뀌어도 문장의 의미가 완전히 다르게 변할 수도 있습니다. 또한, 마찬가지로 띄어쓰기나 어순의 차이로 인한 정제의 이슈도 큰 어려움이 될 수 있습니다. 이러한 어려움을 다루고 해결하기 위한 방법을 이후 챕터에서 다루도록 하겠습니다.