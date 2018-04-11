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
[Structure of Variable. Image from [PyTorch Tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)]

```python
import torch
from torch.autograd import Variable

x = torch.FloatTensor(2, 2)
x = Variable(x, requires_grad = True)

y = torch.FloatTensor(2, 2)
y = Variable(y, requires_grad = False)

z = (x + y) + Variable(torch.FloatTensor(2, 2), requires_grad = True)
```

requires_grad 속성은 직접 생성한 경우에는 False 값을 default로 갖습니다. 연산을 통해 자동으로 생성된 경우(위의 코드 예제에서 z)에는 True 값만 갖도록 됩니다. 따라서 결론적으로 사용자가 지정한 연산/계산을 통해 생성된 computation graph의 leaf node에 해당되는 variable만 requires_grad 값을 True 또는 False로 지정할 수 있습니다. 만약 gradient 자체를 구할 일이 없을 경우(inference 모드, 훈련 중이 아닐 때)에는 volatile 속성을 True 값을 주면 해당 Variable이 속한 computation graph 전체의 gradient를 구하지 않게 됩니다.

## Basic Operation Example (Forward)

```python
a
```

## Extension Class of nn.Module

## Backward and zero_grad()

## train() and eval()

## Using GPU

## Mini-project