# 쉬어가기: MSE 손실 함수와 확률 분포 함수

이번 챕터에서 우리는 뉴럴 네트워크 또한 확률 분포를 정의하는 파라미터 $\theta$ 를 갖는 확률 분포 함수(probability distribution function)이고, 따라서 우리는 우리가 알고자 하는 ground-truth 확률 분포에 근사(approximate)하기 위해서 크로스 엔트로피(cross entropy)를 사용한다고 하였습니다. 하지만 이것은 discrete 랜덤 변수 확률 분포에 국한된다고 하였습니다. Mean square error (MSE) 손실 함수(loss function)를 사용하는 경우는 그럼 무슨 케이스일까요? 사실은 MSE를 사용하는 것은 뉴럴 네트워크가 반환(표현)하는 continuous 확률 분포가 가우시안(gaussian, 정규) 분포를 따른다는 가정이 들어간 것 입니다.

가우시안 분포를 정의하기 위해서는 네트워크 파라미터 $\mu$ (평균)와 $\sigma$ (표준편차)가 필요 합니다. 그러므로 뉴럴 네트워크가 분류 문제(classification problem)에서 softmax 레이어를 활용하여 discrete 확률 분포를 반환(return)했던 것처럼, continuous 확률 분포를 따를 경우(regression problem) $\mu$ 와 $\sigma$ 를 반환하면 가우시안 분포를 반환 할 겁니다. 즉, 뉴럴 네트워크는 샘플 $x_i$ 를 입력으로 받아 가우시안 분포의 $\mu_i$ 와 $\sigma_i$ 를 리턴 할 겁니다. 이것은 마치 아래와 같이 함수꼴로 정의 할 수 있습니다.

$$\mu_i=\mu_\theta(x_i)\text{ and }\sigma_i=\sigma_\phi(x_i)\text{ where }y_i=\mathcal{N}(\mu_i,\sigma_i^2)$$

여기서 $\theta$ 는 함수 $\mu$ 를 정의하는 뉴럴 네트워크 파라미터이고, $\phi$ 는 함수 $\sigma$ 를 정의하는 뉴럴 네트워크 파라미터라고 가정 합니다. 이때, 가우시안 분포의 함수는 아래와 같습니다.

$$\mathcal{N}(\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}\exp{(-\frac{(x-\mu)^2}{2\sigma^2})}$$

이것을 다시 적용하면 우리가 훈련하고자 하는 뉴럴 네트워크 함수 $f$ 는 아래와 같이 표현 가능 합니다.

$$f_{\theta,\phi}(x)=\mathcal{N}\big(\mu_\theta(x), \sigma_\phi(x)^2\big)$$

이제 우리는 negative log-likelihood를 우리의 손실함수(loss function)으로 취하도록 하겠습니다.

$$\begin{aligned}
\mathcal{L}(\theta,\phi)=-\frac{1}{N}\sum_{i=1}^N{\log{f_{\theta,\phi}(x_i)}}
&=-\frac{1}{N}\sum_{i=1}^N{\log{\Big(\frac{1}{\sigma_\phi(x_i)\sqrt{2\pi}}\exp{(-\frac{\big(y_i-\mu_\theta(x_i)\big)^2}{2\sigma_\phi(x_i)^2})}\Big)}}, \\
&=-\frac{1}{N}\sum_{i=1}^N\Big(\log{\frac{1}{\sigma_\phi(x_i)\sqrt{2\pi}}}-\frac{\big(y_i-\mu_\theta(x_i)\big)^2}{2\sigma_\phi(x_i)^2}\Big)
\end{aligned}$$

이때 우리는 가우시안 분포의 $\sigma$ 가 상수라고 가정하겠습니다. 즉, 우리가 관심 있는 값은 오직 $\mu$ 라고 할 때, $\mu$ 에 대해서 미분을 취하면 우리는 마치 아래와 같은 결과를 얻을 수 있습니다.

$$\nabla_\theta\mathcal{L}(\theta,\phi)=\nabla_\theta\Big(\log{\sigma}+\frac{1}{2}\log{2\pi}+\frac{1}{2\sigma\cdot N}\sum_{i=1}^N\big(y_i-\mu_\theta(x_i)\big)^2\Big)$$

즉, 우리의 뉴럴네트워크는 위의 손실함수를 최소화 하도록 그래디언트 디센트를 통해 훈련 될 것 입니다.

$$\begin{gathered}
\hat{\theta}=\underset{\theta}{\text{argmin }}\mathcal{L}(\theta,\phi) \\ \\
\theta \leftarrow \theta-\lambda\nabla_\theta \mathcal{L}(\theta,\phi)
\end{gathered}$$

따라서 새롭게 정의된 손실함수는 아래와 같아질 것 입니다.

$$\nabla_\theta{\mathcal{L}(\theta)}=\nabla_\theta{\frac{1}{2\sigma^2\cdot N}\sum_{i=1}^N\big(y_i-\mu_\theta(x_i)\big)^2}$$

이것은 기존의 MSE와 비교하면 매우 유사함을 알 수 있습니다.

$$\text{MSE}(\theta)=\frac{1}{N}\sum_{i=1}^N\big(y_i-\mu_\theta(x_i)\big)^2$$

즉, 기존의 discrete 확률 분포는 $x$ 로부터 확률 값을 얻을 수 있기 때문에, 크로스 엔트로피를 적용하여 ground-truth 확률 분포에 뉴럴 네트워크 확률 분포를 근사(approximate)할 수 있었습니다. 하지만 continuous 확률 분포는 확률 값을 알 수 없기 때문에, 뉴럴 네트워크의 출력값이 가우시한 분포의 $\mu$ 라는 가정을 하면 MSE를 통해 ground-truth 확률 분포에 뉴럴 네트워크 확률 분포를 근사할 수 있게 되는 것 입니다.
