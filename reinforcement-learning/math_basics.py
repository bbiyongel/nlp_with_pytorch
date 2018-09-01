# Math Basics

## Expectation

![](../assets/lm_rolling_dice.png)

기대값(expectation)은 보상(reward)과 그 보상을 받을 확률을 곱한 값의 총 합을 통해 얻을 수 있습니다. 즉, reward에 대한 가중합(weighted sum)라고 볼 수 있습니다. 주사위의 경우에는 reward의 경우에는 1부터 6까지 받을 수 있지만, 각 reward에 대한 확률은 ${1}/{6}$로 동일합니다.

$$
\begin{aligned}
expected~reward~from~dice&=\sum^6_{x=1}{P(X=x)\times reward(x)} \\
where~P(x)=\frac{1}{6}, \forall x~~&and~~reward(x)=x.
\end{aligned}
$$

따라서 실제 주사위의 기대값은 아래와 같이 3.5가 됩니다.

$$
\frac{1}{6}\times(1+2+3+4+5+6)=3.5
$$

또한, 위의 수식은 아래와 같이 표현 할 수 있습니다.

$$
\mathbb{E}_{X \sim P}[reward(x)]=\sum^6_{x=1}{P(X=x)\times reward(x)}=3.5
$$

주사위의 경우에는 discrete variable을 다루는 확률 분포이고, continuous variable의 경우에는 적분을 통해 우리는 기대값을 구할 수 있습니다.

## Monte Carlo Sampling

Monte Carlo Sampling은 난수를 이용하여 임의의 함수를 근사하는 방법입니다. 예를 들어 임의의 함수 $f$가 있을 때, 사실은 해당 함수가 Gaussian distribution을 따르고 있고, 충분히 많은 수의 random number $x$를 생성하여, $f(x)$를 구한다면, $f(x)$의 분포는 역시 gaussian distribution을 따르고 있을 것 입니다. 이와 같이 임의의 함수에 대해서 Monte Carlo 방식을 통해 해당 함수를 근사할 수 있습니다.

![approximation of pi using Monte Carlo](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)

따라서 Monte Carlo sampling을 사용하면 기대값(expectation) 내의 표현을 밖으로 꺼낼 수 있습니다. 즉, 주사위의 reward에 대한 기대값을 아래와 같이 간단히(simplify) 표현할 수 있습니다.

$$
\mathbb{E}_{X \sim P}[reward(x)] \approx \frac{1}{N}\sum^N_{i=1}{reward(x_i)}
$$

주사위 reward의 기대값은 $N$번 sampling한 주사위 값의 평균이라고 할 수 있습니다. 실제로 $N$이 무한대에 가까워질 수록 (커질 수록) 해당 값은 실제 기대값 $3.5$에 가까워질 것 입니다. 따라서 우리는 경우에 따라서 $N=1$인 경우도 가정 해 볼 수 있습니다. 즉, 아래와 같은 수식이 될 수도 있습니다.

$$
\mathbb{E}_{X \sim P}[reward(x)] \approx reward(x)=x
$$

위와 같은 가정을 가지고 수식을 간단히 표현할 수 있게 되면, 이후 gradient를 구한다거나 할 때에 수식이 간단해져 매우 편리합니다.