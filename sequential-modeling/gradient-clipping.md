# Gradient Clipping

RNN은 BPTT를 통해서 gradient를 구합니다. 따라서 출력의 길이에 따라서 gradient의 크기가 달라지게 됩니다. 즉, 길이가 길수록 자칫 gradient가 너무 커질 수 있기 때문에, learning rate를 조절하는 일이 필요 합니다. 너무 큰 learning rate를 사용하게 되면 gradient descent에서 step의 크기가 너무 커져버려 잘못된 방향으로 학습 및 발산(gradient exloding) 해 버릴 수 있기 때문입니다. 이 경우 가장 쉬운 대처 방법은 learning rate를 아주 작은 값을 취하는 것 입니다. 하지만 작은 learning rate를 사용할 경우, 평소 상황에서 너무 적은 양만 배우므로 훈련 속도가 매우 느려질 것 입니다. 즉, 길이는 가변이므로 learning rate를 그때그때 알맞게 최적의 값을 찾아 조절 해 주는 것은 매우 어려운 일이 될 것입니다. 이때 gradient clipping이 큰 힘을 발휘합니다.

Gradient clipping은 신경망 파라미터 $$\theta$$의 norm (보통 L2 norm)을 구하고, 이 norm의 크기를 제한하는 방법 입니다. 따라서 gradient vector는 방향은 유지하되, 그 크기는 학습이 망가지지 않은 정도로 줄어들 수 있게 됩니다. 물론 norm의 maximum value를 따로 사용자가 정의 해 주어야 하기 때문에, 또 하나의 hyper-parameter가 생기게 되지만, 큰 norm을 가진 gradient vector의 경우에만 gradient clipping을 수행하기 때문에, 능동적으로 learning rate를 조절하는 것과 비슷한 효과를 가질 수 있습니다. 따라서 gradient clipping은 RNN 계열의 학습 및 훈련을 할 때 널리 사용되는 방법 입니다.

$$
\begin{aligned}
\frac{\partial\epsilon}{\partial\theta} \leftarrow &\begin{cases}
   \frac{\text{threshold}}{\Vert\hat{g}\Vert}\hat{g} &\text{if } \Vert\hat{g}\Vert\ge\text{threshold}  \\
   \hat{g} &\text{otherwise}
\end{cases} \\
&\text{where }\hat{g}=\frac{\partial\epsilon}{\partial\theta}.
\end{aligned}
$$

수식을 보면, gradient norm이 정해진 threshold(역치)보다 클 경우에, gradient 벡터를 threshold 보다 큰 만큼의 비율로 나누어 줍니다. 따라서 gradient는 항상 threshold 보다 작으며 이는 gradient exploding을 방지함과 동시에, gradient의 방향을 유지해주기 때문에 모델 파라미터 $$\theta$$가 학습해야 하는 방향은 잃지 않습니다.

PyTorch에서도 기능을 [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/nn.html?highlight=clip#torch.nn.utils.clip_grad_norm_) 이라는 함수를 제공하고 있으므로 매우 쉽게 사용 할 수 있습니다.