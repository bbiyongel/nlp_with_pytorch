# Monte Carlo

## 소개

## 수식

# Policy Gradient

Policy Gradient는 ***Policy based Reinforcement Learning*** 방식입니다. 알파고를 개발했던 DeepMind에 의해서 유명해진 Deep Q-Learning은 Value based Reinforcement Learning 방식에 속합니다. 실제 딥러닝을 사용하여 두 방식을 사용 할 때에 가장 큰 차이점은, Value based방식은 인공신경망을 사용하여 어떤 action을 하였을 때에 얻을 수 있는 보상을 예측 하도록 훈련하는 것과 달리, policy based 방식은 인공신경망은 어떤 action을 할지 훈련되고 해당 action에 대한 보상(reward)를 back-propagation 할 때에 gradient를 통해서 전달해 주는 것이 가장 큰 차이점 입니다. 따라서 어떤 Deep Q-learning의 경우에는 action을 선택하는 것이 deterministic한 것에 비해서, ***Policy Gradient***방식은 action을 선택 할 때에 stochastic한 process를 거치게 됩니다. Policy Gradient에 대한 수식은 아래와 같습니다.

$$
\pi_\theta(a|s) = P_\theta(a|s) = P(a|s; \theta)
$$
$$
J(\theta) = v_{\pi_\theta}(S_0)~and~we~need~to~maximize~J(\theta)
$$
$$
\theta_{t+1}=\theta_t+\alpha\triangledown_\theta J(\theta)
$$
$$
\triangledown_\theta J(\theta) = \triangledown_\theta v_{\pi_\theta}(S_0)
$$
$$
\triangledown_\theta J(\theta) = E_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)}q_\pi (s, a)]
$$

위의 수식을 NLP에 적용하여 쉽게 설명 해 보면, Monte Carlo 방식을 통해 sampling 된 sentence에 대해서 gradient를 구하고, 그 gradient에 reward를 곱하여 주는 형태입니다. 만약 샘플링 된 해당 문장이 좋은 (큰 양수) reward를 받았다면 learning rate $$ \alpha $$에 추가적인 scaling을 통해서 더 큰 step으로 gradient ascending을 할 수 있을 겁니다. 하지만 negative reward를 받게 된다면, gradient는 반대로 적용이 되도록 값이 곱해지게 될 겁니다. 따라서 해당 샘플링 된 문장이 나오지 않도록 parameter $$ \theta $$가 update 될 겁니다.

Policy Gradient에 대한 자세한 설명은 원 논문인 [[Sutton at el.1999]](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), 또는 해당 저자가 쓴 textbook ['Reinforcement Learning: An Introduction'](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf)을 참고하거나, [DeepMind David Silver의 YouTube 강의](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)를 참고하면 좋습니다. 