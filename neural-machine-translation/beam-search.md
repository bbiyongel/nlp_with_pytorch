# Inference

# Overview

이제까지 $$ X $$와 $$ Y $$가 모두 주어진 훈련상황을 가정하였습니다만, 이제부터는 $$ X $$만 주어진 상태에서 $$ \hat{Y} $$을 예측하는 방법에 대해서 서술하겠습니다. 이러한 과정을 우리는 Inference 또는 Search 라고 부릅니다. 우리가 기본적으로 이 방식을 search라고 부르는 이유는 search 알고리즘에 기반하기 때문입니다. 결국 우리가 원하는 것은   state로 이루어진 단어(word) 사이에서 최고의 확률을 갖는 path를 찾는 것이기 때문입니다.

# 1. Sampling

사실 먼저 우리가 생각할 수 있는 가장 정확한 방법은 각 time-step별 $$ \hat{y}_t $$를 고를 때, 마지막 ***softmax*** layer에서의 확률 분포(probability distribution)대로 sampling을 하는 것 입니다. 그리고 다음 time-step에서 그 선택($$ \hat{y}_t $$)을 기반으로 다음 $$ \hat{y}_{t+1} $$을 또 다시 sampling하여 최종적으로 $$ EOS $$가 나올 때 까지 sampling을 반복하는 것 입니다. 이렇게 하면 우리가 원하는 $$ P(Y|X) $$ 에 가장 가까운 형태의 번역이 완성될 겁니다. 하지만, 이러한 방식은 같은 입력에 대해서 매번 다른 출력 결과물을 만들어낼 수 있습니다. 따라서 우리가 원하는 형태의 결과물이 아닙니다.

# 2. Gready Search

우리는 자료구조, 알고리즘 수업에서 수 많은 search 방법에 대해 배웠습니다. DFS, BFS, Dynamic Programming 등. 우리는 이 중에서 Greedy algorithm을 기반으로 search를 구현합니다. 즉, softmax layer에서 가장 값이 큰 index를 뽑아 해당 time-step의 $$ \hat{y}_t $$로 사용하게 되는 것 입니다.

## Beam Search

![](/assets/beam_search.png)

하지만 우리는 자료구조, 알고리즘 수업에서 배웠다시피, greedy algorithm은 굉장히 쉽고 간편하지만, 최적의(optimal) 해를 보장하지 않습니다. 따라서 최적의 해에 가까워지기 위해서 우리는 약간의 trick을 첨가합니다. ***Beam Size*** 만큼의 후보를 더 tracking 하는 것 입니다.

현재 time-step에서 Top-***k***개를 뽑아서 (여기서 k는 beam size와 같습니다) 다음 time-step에 대해서 k번 inference를 수행합니다. 그리고 총 $$ k * |V| $$ 개의 softmax 결과 값 중에서 다시 top-k개를 뽑아 다음 time-step으로 넘깁니다. ($$ |V| $$는 Vocabulary size) 여기서 중요한 점은 두가지 입니다.

$$
\hat{y}_{t}^{k} = argmax_{k\text{-}th} \hat{Y}_t
$$
$$
\hat{Y}_{t} = f_\theta(X, y_{<t}^{1}) \cup f_\theta(X, y_{<t}^{2}) \cup \cdots \cup f_\theta(X, y_{<t}^{k})
$$
$$
X=\{x_1, x_2, \cdots, x_n\}
$$

1. 누적 확률을 사용하여 top-k를 뽑습니다. 이때, 보통 로그 확률을 사용하므로 현재 time-step 까지의 로그확률에 대한 합을 tracking 하고 있어야 합니다.
2. top-k를 뽑을 때, 현재 time-step에 대해 k번 계산한 모든 결과물 중에서 뽑습니다.

Beam Search를 사용하면 좀 더 넓은 path에 대해서 search를 수행하므로 당연히 좀 더 나은 성능을 보장합니다. 하지만, beam size만큼 번역을 더 수행해야 하기 때문에 속도에 저하가 있습니다. 다행히도 우리는 이 작업을 mini-batch로 만들어 수행하기 때문에, 병렬처리로 인해서 약간의 속도저하만 생기게 됩니다.

아래는 [Cho et al.2016]에서 주장한 ***Beam Search***의 성능향상에 대한 실험 결과 입니다. Sampling 방법은 단순한 Greedy Search 보다 더 좋은 성능을 제공하지만, Beam search가 가장 좋은 성능을 보여줍니다. 특기할 점은 Machine Translation task에서는 보통 beam size를 10이하로 사용한다는 것 입니다. 

![http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture10.pdf](/assets/decoding_performance.png)
En-Cz: 12m training sentence pairs [Cho, arXiv 2016]

## Length Penalty

위의 search 알고리즘을 직접 짜서 수행시켜 보면 한가지 문제점이 발견됩니다. 현재 time-step 까지의 확률을 모두 곱(로그확률의 경우에는 합)하기 때문에 문장이 길어질 수록 확률이 낮아진다는 점 입니다. 따라서 짧은 문장일수록 더 높은 점수를 획득하는 경향이 있습니다. 우리는 이러한 현상을 방지하기 위해서 ***length penalty***를 주어 search가 조기 종료되는 것을 막습니다.

수식은 아래와 같습니다. 불행히도 우리는 몇개의 hyper-parameter를 추가해야 합니다.

$$
p = \frac{(1 + length)^\alpha}{(1 + min\text{_}length)^\alpha}
$$