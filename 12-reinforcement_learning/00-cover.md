# Reinforcement Learning for Natural Language Processing

![Richard S. Sutton: Professor at University of Alberta](../assets/12-00-01.jpeg){ width=500px }

자연어생성 문제는 시퀀셜 데이터 생성의 특성상 과거 자신의 상태에 따라 미래의 상태가 결정되는 auto-regressive한 속성을 가지게 됩니다. 따라서 기존의 뉴럴 네트워크 훈련 방식인 Teacher-forcing과 Maximum Likelihood Estimation(MLE)으로는 정확한 훈련을 수행할 수 없었습니다. 물론 기본적인 훈련 방법을 통해서도 기존의 수십년의 연구성과를 훌쩍 뛰어넘는 결과를 만들어낼 수 있었지만, 한편으로는 정확하지 못한 훈련 방법으로 인한 아쉬움이 남아있었습니다. 하지만 우리는 강화학습을 활용하여 샘플링을 통해 이 문제를 해결하고 좀 더 정확한 자연어 문장을 생성할 수 있습니다. 이번 챕터에서는 자연어생성을 강화학습에 적용하기 위하여, 강화학습의 기초적인 내용 전반에 대해 흝어보고, 강화학습을 통해서 자연어 생성 문제의 성능을 향상시킬 수 있는 방법과 실제 문제 적용에 대해서 다루고자 합니다. 또한 실제 샘플링 기반의 강화학습 수식을 실제 파이토치 코드로 어떻게 구현하는지 예제 코드를 통해 살펴보도록 하겠습니다. 이를 통해 독자분들이 실제 수학적으로 이끌어낸 수식이 코드로 어떻게 바뀌어 적용되는지 알 수 있었으면 합니다.
