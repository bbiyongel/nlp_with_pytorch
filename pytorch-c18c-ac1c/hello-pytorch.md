# Hello, PyTorch

## Tensor

PyTorch의 tensor는 numpy의 array와 같은 개념입니다. 값을 저장하고 그 값들에 대해서 연산을 수행할 수 있는 함수를 제공합니다.

```python
import torch

x = torch.FloatTensor(3, 3)
```

## Variable and Autograd

자동으로 gradient를 계산할 수 있게 하기 위해서, tensor를 wrapping해 주는 class입니다. Variable은 gradient를 저장할 수 있는 **grad**와 tensor를 저장하는 **data** 속성(attribute)을 갖고 있습니다.

![](http://pytorch.org/tutorials/_images/Variable.png)

## Basic Operation (Forward)

## Backward

## Extension Class of nn.Module

## Mini-project