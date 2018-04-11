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

자동으로 gradient를 계산할 수 있게 하기 위해서, tensor를 wrapping해 주는 class입니다. Variable은 gradient를 저장할 수 있는 **grad**와 tensor를 저장하는 **data** 속성\(attribute\)을 갖고 있습니다. 또한 **grad\_fn**이라는 속성은 variable을 생성한 연산\(또는 함수\)를 가리키고 있어, 연산\(feed-forward\)에 따라 마지막까지 자동으로 생성된 Variable을 사용하여 최초 계산에 사용된 Variable까지의 gradient를 자동으로 계산 해 줍니다.

![](http://pytorch.org/tutorials/_images/Variable.png)  
\[Structure of Variable. Image from [PyTorch Tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)\]

requires\_grad 속성은 직접 생성한 경우에는 False 값을 default로 갖습니다. 연산을 통해 자동으로 생성된 경우\(위의 코드 예제에서 z\)에는 True 값만 갖도록 됩니다. 따라서 결론적으로 사용자가 지정한 연산/계산을 통해 생성된 computation graph의 leaf node에 해당되는 variable만 requires\_grad 값을 True 또는 False로 지정할 수 있습니다. 만약 gradient 자체를 구할 일이 없을 경우\(inference 모드, 훈련 중이 아닐 때\)에는 volatile 속성을 True 값을 주면 해당 Variable이 속한 computation graph 전체의 gradient를 구하지 않게 됩니다.

```python
import torch
from torch.autograd import Variable

x = torch.FloatTensor(2, 2)
x = Variable(x, requires_grad = True)

y = torch.FloatTensor(2, 2)
y = Variable(y, requires_grad = False)

z = (x + y) + Variable(torch.FloatTensor(2, 2), requires_grad = True)
```

위의 코드에서 x와 y를 Variable로 선언하고 더한 후에, 변수로 지정하지 않은 Variable을 더하고 그 값을 z에 저장합니다. 따라서 아래와 같은 computation graph를 가지게 됩니다. x, y, z는 leaf node에 해당하므로 requires_grad를 사용자가 임의로 설정할 수 있습니다. 이후에 z에 gradient가 전달되어 오면, 연산 과정에서 형성된 tree 구조를 통해 chide node들에게 gradient를 전달 할 수 있습니다.

![](/assets/pytorch-intro-xyz-graph.png)

## Basic Operation Example \(Forward\)

$$
y = xW^t + b
$$

```python
import torch
from torch.autograd import Variable

def linear(x):
    W = Variable(torch.FloatTensor(10, 5), requires_grad = True)
    b = Variable(torch.FloatTensor(5), requires_grad = True)
    
    y = torch.mm(x, W) + b
    
    return y

x = torch.FloatTensor(16, 10)
x = Variable(x)

y = linear(x)
```

### Broadcasting

```python
>>> # vector x vector
>>> tensor1 = torch.randn(3)
>>> tensor2 = torch.randn(3)
>>> torch.matmul(tensor1, tensor2).size()

-0.4334
[torch.FloatTensor of size ()]

>>> # matrix x vector
>>> tensor1 = torch.randn(3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([3])
>>> # batched matrix x broadcasted vector
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3])
>>> # batched matrix x batched matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(10, 4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
>>> # batched matrix x broadcasted matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
```

참고사이트: [http://pytorch.org/docs/master/torch.html?highlight=matmul\#torch.matmul](http://pytorch.org/docs/master/torch.html?highlight=matmul#torch.matmul)

## Extension Class of nn.Module

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

class MyLinear(nn.Module):

    def __init__(self.input_size, output_size):
        super(MyLinear, self).__init__()
        
        self.W = Variable(torch.FloatTensor(input_size, output_size), requires_grad = True)
        self.b = Variable(torch.FloatTensor(output_size), requires_grad = True)
        
    def forward(self.x):
        y = torch.mm(x, self.W) + self.b
        
        return y
```

```python        
>>> x = torch.FloatTensor(16, 10)
>>> x = Variable(x)
>>> linear = MyLinear(10, 5)
>>> y = linear(x)
```

```python
class MyLinear(nn.Module):

    def __init__(self.input_size, output_size):
        super(MyLinear, self).__init__()
        
        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size), requires_grad = True)
        self.b = nn.Parameter(torch.FloatTensor(output_size), requires_grad = True)
        
    def forward(self.x):
        y = torch.mm(x, self.W) + self.b
        
        return y
```

참고사이트: http://pytorch.org/docs/master/nn.html?highlight=parameter#parameters

>when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model.

```python
>>> print(linear)
MyLinear(
)
```

```python
class MyLinear(nn.Module):

    def __init__(self.input_size, output_size):
        super(MyLinear, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
                
    def forward(self.x):
        y = self.linear(x)
        
        return y
```

```python
>>> print(linear)
MyLinear(
  (linear): Linear(in_features=10, out_features=5, bias=True)
)
```

## Backward and zero\_grad\(\)

$$
\begin{aligned}
&\mathcal{L}_{MSE}(x, y)=\frac{1}{N}\sum^N_{i=1}{(x_n - y_n)^2}
\end{aligned}
$$

## train\(\) and eval\(\)

```python
# Training...
linear.eval()
# Do some inference process.
linear.train()
# Restart training, again.
```

## Using GPU

## Mini-project



