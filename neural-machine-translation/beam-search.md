# Inference

# Overview

이제까지 $$ X $$와 $$ Y $$가 모두 주어진 훈련상황을 가정하였습니다만, 이제부터는 $$ X $$만 주어진 상태에서 $$ \hat{Y} $$을 예측하는 방법에 대해서 서술하겠습니다. 이러한 과정을 우리는 Inference 또는 Search 라고 부릅니다. 우리가 기본적으로 이 방식을 search라고 부르는 이유는 search 알고리즘에 기반하기 때문입니다. 결국 우리가 원하는 것은   state로 이루어진 단어(word) 사이에서 최고의 확률을 갖는 path를 찾는 것이기 때문입니다.

# 1. Sampling

사실 먼저 우리가 생각할 수 있는 가장 정확한 방법은 각 time-step별 $$ \hat{y}_t $$를 고를 때, 마지막 ***softmax*** layer에서의 확률 분포(probability distribution)대로 sampling을 하는 것 입니다. 그리고 다음 time-step에서 그 선택($$ \hat{y}_t $$)을 기반으로 다음 $$ \hat{y}_{t+1} $$을 또 다시 sampling하여 최종적으로 $$ EOS $$가 나올 때 까지 sampling을 반복하는 것 입니다. 이렇게 하면 우리가 원하는 $$ P(Y|X) $$ 에 가장 가까운 형태의 번역이 완성될 겁니다. 하지만, 이러한 방식은 같은 입력에 대해서 매번 다른 출력 결과물을 만들어낼 수 있습니다. 따라서 우리가 원하는 형태의 결과물이 아닙니다.

# 2. Gready Search

우리는 자료구조, 알고리즘 수업에서 수 많은 search 방법에 대해 배웠습니다. DFS, BFS, Dynamic Programming 등. 우리는 이 중에서 Greedy algorithm을 기반으로 search를 구현합니다. 즉, softmax layer에서 가장 값이 큰 index를 뽑아 해당 time-step의 $$ \hat{y}_t $$로 사용하게 되는 것 입니다.

## Beam Search

![](/assets/beam_search.png)

하지만 우리는 자료구조, 알고리즘 수업에서 배웠다시피, greedy algorithm은 굉장히 쉽고 간편하지만, 최적의(optimal) 해를 보장하지 않습니다. 따라서 최적의 해에 가까워지기 위해서 우리는 약간의 trick을 첨가합니다.

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/decoding_performance.png)
En-Cz: 12m training sentence pairs [Cho, arXiv 2016]

## Length Penelty