# GloVe

이전 섹션에서는 Word2Vec에 대해서 다루어 보았습니다. 이번 섹션에서는 또 하나의 대표적인 워드 임베딩 방법인 GloVe (Global Vectors for Word Representation) [Pennington et al.2014]에 대해 다루어보고자 합니다. 

## 알고리즘

이전에 다루었던 skip-gram은 대상 단어를 통해 주변 단어를 예측하도록 네트워크를 구성하여 단어 임베딩 벡터를 학습하였습니다. GloVe는 대신에 대상 단어에 대해서 코퍼스(corpus)에 함꼐 나타난 각 단어별 출현 빈도를 예측하도록 합니다. GloVe 알고리즘의 네트워크 파라미터를 구하는 수식은 아래와 같습니다.

$$
\begin{gathered}
\hat{\theta}=\underset{\theta}{\text{argmin}}\sum_{x\in\mathcal{X}}f(x)\times\big|W'Wx-\log{C_x}\big|_2 \\
\text{where }C_x\text{ is vector of co-occurences with }x \\ 
\text{Also, }x\in\{0,1\}^{|V|}, W\in\mathbb{R}^{d\times|V|}\text{ and }W'\in\mathbb{R}^{|V|\times d}.
\end{gathered}
$$

Skip-gram을 위한 네트워크와 거의 유사한 형태임을 알 수 있습니다. 다만, 여기서는 분류(classification) 문제 <comment> Skip-gram은 마지막 레이어가 소프트맥스로 구성되어 있었습니다. </comment>가 아닌, 출현 빈도를 근사(approximation)하는 리그레션(regression)문제에 가깝기 때문에, Mean Square Error (MSE)를 사용한 것을 볼 수 있습니다. 

마찬가지로 one-hot 인코딩 벡터 $x$를 입력으로 받아 한 개의 히든 레이어 $W$를 거쳐 출력 레이어 $W'$를 통해 출력 벡터를 반환 합니다. 이 출력 벡터는 단어 $x$와 함께 코퍼스에 출현했던 모든 단어들의 각 동시 출현 빈도들을 나타낸 벡터인 $C_x$를 근사해야 합니다. 따라서 이 둘의 차이값인 손실(loss)를 최소화 하도록 back-propagation 및 그래디언트 디센트를 통해 학습을 할 수 있습니다.

이때 단어 $x$ 자체의 출현빈도 또는 사전확률(prior probability)에 따라서 MSE 손실함수의 값이 매우 달라질 것 입니다. 예를 들어, $C_x$가 클수록 손실값은 커질것이기 때문입니다. 따라서 $f(x)​$는 단어의 빈도에 따라 아래와 같이 손실 함수에 가중치를 부여합니다.

$$
f(x)=
\begin{cases}
\big({\text{Count}(x)}/{\text{thres}}\big)^\alpha & \text{Count}(x)<\text{thres} \\
1 & \text{otherwise}.
\end{cases}
$$

이 논문에서는 실험에 의해 $\text{thres}=100$, $\alpha=3/4$ 일때 가장 좋은 결과가 나온다고 언급하였습니다.

## 장점

코퍼스 내에서 주변단어를 예측하고자 하는 Skip-gram과 달리, GloVe는 처음에 코퍼스를 통해 각 단어별 동시 출현 빈도를 조사하여 이에 대한 출현빈도 행렬(matrix)을 만들고, 이후엔 해당 행렬을 통해 동시 출현 빈도를 근사(approximate)하고자 합니다. 따라서 코퍼스 전체를 흝으며 대상 단어와 주변 단어를 가져와 학습하는 과정을 반복해야 하는 skip-gram과 달리 훨씬 학습이 빠릅니다.

또한, 코퍼스를 흝으며 학습하는 skip-gram의 특성상, 사전확률(prior probability)이 작은 (즉, 출현 빈도 자체가 작은) 단어에 대해서는 학습 기회가 적을 수 밖에 없습니다. 따라서 출현 빈도가 작은 단어들은 비교적 부정확한 단어 임베딩 벡터를 학습하게 됩니다. 하지만, GloVe는 skip-gram에 비해서 이러한 단점이 완벽하진 않지만 어느정도 보완되었습니다.

## 결론

비록 GloVe의 저자는 GloVe가 가장 뛰어난 임베딩 방식임을 주장하였지만, 사실 skip-gram도 파라미터 <comment> 윈도우 사이즈와 러닝레이트 학습 이터레이션 수 등 </comment> 튜닝 여부에 따라 GloVe와 큰 성능 차이가 없습니다. 따라서 실제 구현하고자 할때에는 주어진 상황<comment> 예를 들어 시스템 구성에 대한 제약 </comment>에 따라 적절한 방법을 선택하는 것도 한가지 방법입니다.
