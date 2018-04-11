# Hello, PyTorch

## Tensor

PyTorch의 tensor는 numpy의 array와 같은 개념입니다. 값을 저장하고 그 값들에 대해서 연산을 수행할 수 있는 함수를 제공합니다.

```python
import torch

x = torch.FloatTensor(2, 2)
x = torch.FloatTensor([[1, 2], [3, 4]])

import numpy as np

x = [[1, 2], [3, 4]]
x = np.array(x)
x = torch.from_numpy(x)
```

## Variable and Autograd

자동으로 gradient를 계산할 수 있게 하기 위해서, tensor를 wrapping해 주는 class입니다. Variable은 gradient를 저장할 수 있는 **grad**와 tensor를 저장하는 **data** 속성(attribute)을 갖고 있습니다. 또한 **grad_fn**이라는 속성은 variable을 생성한 연산(또는 함수)를 가리키고 있어, 연산(feed-forward)에 따라 마지막까지 자동으로 생성된 Variable을 사용하여 최초 계산에 사용된 Variable까지의 gradient를 자동으로 계산 해 줍니다.

![](http://pytorch.org/tutorials/_images/Variable.png)

```python
import torch
from torch.autograd import Variable

x = torch.FloatTensor(2, 2)
x = Variable(x, requires_grad = True)

y = torch.FloatTensor(2, 2)
y = Variable(y, requires_grad = False)

z = (x + y) + Variable(torch.FloatTensor(2, 2), requires_grad = True)
```

## Basic Operation (Forward)

## Backward

## Extension Class of nn.Module

## train() and eval()

## Using GPU

## Mini-project