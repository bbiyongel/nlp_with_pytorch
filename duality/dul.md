# Dual Unsupervised Learning
  
공교롭게도 비슷한 시기에 나온 논문[\[Xia at el.2016\]](https://arxiv.org/pdf/1611.00179.pdf)이 있습니다. ***GAN***이 안되는 NLP의 특성상 CycleGAN처럼 direct로 gradient를 이어줄 수는 없었지만 기본적으로는 아주 비슷한 idea입니다. 짝이 없는 monolingual corpus를 이용하여 성능을 극대화 하고자 하였습니다.

즉, monolingual sentence($$ s $$)에 대해서 번역을 하고 그 문장($$ s_{mid} $$)을 사용하여 복원을 하였을 때($$ \hat{s} $$) 원래의 처음 문장으로 돌아올 수 있도록(처음 문장과의 차이를 최소화 하도록) 훈련하는 것입니다. 이때, 번역된 문장 $$ s_{mid} $$는 자연스러운 해당 언어의 문장이 되었는가도 중요한 지표가 됩니다.

![](/assets/rl-dual-learning-1.png)

위에서 설명한 algorithm을 따라가 보겠습니다. 이 방법에서는 $$ Set~X,~Set~Y $$ 대신에 $$ Language~A,~Language~B $$로 표기하고 있습니다. $$ G_{A \rightarrow B} $$의 파라미터 $$ \theta_{AB} $$와 $$ F_{B \rightarrow A} $$의 파라미터 $$ \theta_{BA} $$가 등장합니다. 이 $$ G_{A \rightarrow B}, F_{B \rightarrow A} $$는 모두 parallel corpus에 의해서 pre-training이 되어 있는 상태 입니다. 즉, 기본적인 저성능의 번역기 수준이라고 가정합니다.

우리는 기존의 policy gradient와 마찬가지로 아래와 같은 parameter update를 수행해야 합니다.

$$
\theta_{AB} \leftarrow \theta_{AB} + \gamma \triangledown_{\theta_{AB}}\hat{E}[r]
$$
$$
\theta_{BA} \leftarrow \theta_{BA} + \gamma \triangledown_{\theta_{BA}}\hat{E}[r]
$$

$$ \hat{E}[r] $$을 각각의 parameter에 대해서 미분 해 준 값을 더해주는 것을 볼 수 있습니다. 이 reward의 기대값은 아래와 같이 구할 수 있습니다.

$$
r=\alpha r_{AB} + (1-\alpha)r_{BA}
$$
$$
r_{AB}=LM_{B}(s_{mid})
$$
$$
r_{BA}=\log{P(s|s_{mid};\theta_{BA})}
$$

위와 같이 $$ k $$개의 sampling한 문장에 대해서 각기 방향에 대한 reward를 각각 구한 후, 이를 linear combination을 취해줍니다. 이때, $$ s_{mid} $$는 sampling한 문장을 의미하고, $$ LM_B $$를 사용하여 해당 문장이 $$ language B $$의 집합에 잘 어울리는지를 따져 reward로 리턴합니다. 여기서 기존의 cross entropy를 사용할 수 없는 이유는 monlingual sentence이기 때문에 번역을 하더라도 정답을 알 수 없기 때문입니다. 또한 우리는 다수의 monolingual corpus를 갖고 있기 때문에, $$ LM $$은 쉽게 만들어낼 수 있습니다.

$$
\triangledown_{\theta_{AB}}\hat{E}[r]=\frac{1}{K}\sum_{k=1}^K{[r_k\triangledown_{\theta_{AB}}\log{P(s_{mid,k}|s;\theta_{AB})}]}
$$
$$
\triangledown_{\theta_{BA}}\hat{E}[r]=\frac{1}{K}\sum_{k=1}^K[(1-\alpha)\triangledown_{\theta_{BA}}\log{P(s|s_{mid,k};\theta_{BA})}]
$$

이렇게 얻어진 $$ E[r] $$를 각 parameter에 대해서 미분하게 되면 위와 같은 수식을 얻을 수 있고, 상기 서술한 parameter update 수식에 대입하면 됩니다. 비슷한 방식으로 $$ B \rightarrow A $$를 구할 수 있습니다.

![](/assets/rl-dual-learning-2.png)

위의 table은 이 방법의 성능을 비교한 결과 입니다. Pseudo-NMT는 이전 챕터에서 설명하였던 back-translation을 의미합니다. 그리고 그 방식보다 더 좋은 성능을 기록한 것을 볼 수 있습니다.

![](/assets/rl-dual-learning-3.png)

또한, 위 그래프에서 문장의 길이와 상관 없이 모든 구간에서 baseline NMT를 성능으로 압도하고 있는 것을 알 수 있습니다. 다만, parallel corpus의 양이 커질수록 monolingual corpus에 의한 성능 향상의 폭이 줄어드는 것을 확인 할 수 있습니다.

