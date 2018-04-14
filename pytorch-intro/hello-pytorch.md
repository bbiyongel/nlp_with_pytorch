# Hello PyTorch,

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

## How to Do Basic Operations \(Forward\)

이번에는 Linear Layer(또는 fully-connected layer, dense layer)를 구현 해 보도록 하겠습니다. M by N의 입력 matrix가 주어지면, N by P의 matrix를 곱한 후, P size의 vector를 bias로 더하도록 하겠습니다. 수식은 아래와 같을 것 입니다.

$$
y = xW^t + b
$$

이러한 linaer layer의 기능은 아래와 같이 PyTorch로 구현할 수 있습니다.

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

PyTorch에 새롭게 추가된 기능인 Broadcasting에 대해서 설명 해 보겠습니다. NumPy에서 제공되는 broadcasting과 동일하게 동작합니다. **matmul()**을 사용하면 임의의 차원의 tensor끼리 연산을 가능하게 해 줍니다. 이전에는 강제로 2차원을 만들거나 하여 곱해주는 수 밖에 없었습니다. 다만, 입력으로 주어지는 tensor들의 차원에 따라서 규칙이 적용됩니다. 그 규칙은 아래와 같습니다.

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

마찬가지로 덧셈 연산에 대해서도 broadcasting이 적용될 수 있는데 그 규칙은 아래와 같습니다. 곱셈에 비해서 좀 더 규칙이 복잡하니 주의해야 합니다.

```python
>>> x=torch.FloatTensor(5,7,3)
>>> y=torch.FloatTensor(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

>>> x=torch.FloatTensor()
>>> y=torch.FloatTensor(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
>>> x=torch.FloatTensor(5,3,4,1)
>>> y=torch.FloatTensor(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
>>> x=torch.FloatTensor(5,2,4,1)
>>> y=torch.FloatTensor(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

# can line up trailing dimensions to make reading easier
>>> x=torch.FloatTensor(5,1,4,1)
>>> y=torch.FloatTensor(  3,1,1)
>>> (x+y).size()
torch.Size([5, 3, 4, 1])

# but not necessary:
>>> x=torch.FloatTensor(1)
>>> y=torch.FloatTensor(3,1,7)
>>> (x+y).size()
torch.Size([3, 1, 7])

>>> x=torch.FloatTensor(5,2,4,1)
>>> y=torch.FloatTensor(3,1,1)
>>> (x+y).size()
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```

Broadcasting 연산의 가장 주의해야 할 점은, 의도하지 않은 broadcasting연산으로 인해서 bug가 발생할 가능성 입니다. 원래는 같은 size의 tensor끼리 연산을 해야 하는 부분인데, 코딩하며 실수에 의해서 다른 size가 되었을 때, 덧셈 또는 곱셈을 하고 error가 나서 알아차려야 하지만, error가 나지 않고 넘어가 버린 상태에서, 결국 기대하던 값과 다른 값이 결과로 나오게 되어, 원인을 찾느라 고생할 수도 있습니다. 따라서 코딩 할 때에 요령이 필요합니다.

> 참고사이트: 
> - [http://pytorch.org/docs/master/torch.html?highlight=matmul\#torch.matmul](http://pytorch.org/docs/master/torch.html?highlight=matmul#torch.matmul)
> - http://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics

## nn.Module

이제까지 우리가 원하는 수식을 어떻게 어떻게 feed-forward 구현 하는지 살펴 보았습니다. 이것을 좀 더 편리하고 깔끔하게 사용하는 방법에 대해서 다루어 보도록 하겠습니다. PyTorch는 nn.Module이라는 class를 제공하여 사용자가 이 위에서 자신이 필요로 하는 model architecture를 구현할 수 있도록 하였습니다. 

nn.Module의 상속한 사용자 정의 class는 다시 내부에 nn.Module을 상속한 class를 선언하여 소유 할 수 있습니다. 즉, nn.Module 안에 nn.Module 객체를 선언하여 사용 할 수 있습니다. 그리고 nn.Module의 forward() 함수를 override하여 feed-forward를 구현할 수 있습니다. 이외에도 nn.Module의 특성을 이용하여 한번에 weight parameter를 save/load할 수도 있습니다.

그럼 앞서 구현한 linear 함수 대신에 MyLinear라는 class를 nn.Module을 상속받아 선언하고, 사용하여 똑같은 기능을 구현 해 보겠습니다.

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

class MyLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()
        
        self.W = Variable(torch.FloatTensor(input_size, output_size), requires_grad = True)
        self.b = Variable(torch.FloatTensor(output_size), requires_grad = True)
        
    def forward(self, x):
        y = torch.mm(x, self.W) + self.b
        
        return y
```

위와 같이 선언한 MyLinear class를 이제 직접 사용해서 정상 동작 하는지 확인 해 보겠습니다.

```python        
x = torch.FloatTensor(16, 10)
x = Variable(x)
linear = MyLinear(10, 5)
y = linear(x)
```

**forward()**에서 정의 해 준대로 잘 동작 하는 것을 볼 수 있습니다. 하지만, 위와 같이 W와 b를 선언하면 문제점이 있습니다. parameters() 함수는 module 내에 선언 된 learnable parameter들을 iterative하게 주는 iterator를 반환하는 함수 입니다. 한번, linear module 내의 learnable parameter들의 크기를 size()함수를 통해 확인 해 보도록 하겠습니다.

```python
>>> params = [p.size() for p in linear.parameters()]
>>> print(params)
[]
```

아무것도 들어있지 않은 빈 list가 찍혔습니다. 즉, linear module 내에는 learnable parameter가 없다는 이야기 입니다. 그 이유는 __init__() 내에서 Variable로 선언 하였기 때문 입니다. 아래의 웹페이지에 그 이유가 자세히 나와 있습니다.

참고사이트: http://pytorch.org/docs/master/nn.html?highlight=parameter#parameters

>when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model.

따라서 우리는 Variable 대신에 Parameter라는 class를 사용하여 tensor를 wrapping해야 합니다. 그럼 아래와 같이 될 것 입니다.

```python
class MyLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()
        
        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size), requires_grad = True)
        self.b = nn.Parameter(torch.FloatTensor(output_size), requires_grad = True)
        
    def forward(self, x):
        y = torch.mm(x, self.W) + self.b
        
        return y
```

그럼 아까와 같이 다시 linear module 내부의 learnable parameter들의 size를 확인 해 보도록 하겠습니다.

```python
>>> params = [p.size() for p in linear.parameters()]
>>> print(params)
[torch.Size([10, 5]), torch.Size([5])]
```

잘 들어있는 것을 확인 할 수 있습니다. 그럼 아래와 같이 한번 실행 해 보죠.

```python
>>> print(linear)
MyLinear(
)
```

아쉽게도 Parameter로 선언 된 parameter들은 print로 찍혀 나오지 않습니다. -- 왜 그렇게 구현 해 놓았는지 이유는 잘 모르겠습니다. 그럼 print에서도 확인할 수 있게 깔끔하게 바꾸어 보도록 하겠습니다. 아래와 같이 바꾸면 제대로 된 구현이라고 볼 수 있습니다.

```python
class MyLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
                
    def forward(self, x):
        y = self.linear(x)
        
        return y
```

nn.Linear class를 사용하여 W와 b를 대체하였습니다. 그리고 아래와 같이 print를 해 보면 내부의 Linear Layer가 잘 찍혀 나오는 것을 확인 할 수 있습니다.

```python
>>> print(linear)
MyLinear(
  (linear): Linear(in_features=10, out_features=5, bias=True)
)
```

## Backward (Back-propagation)

이제까지 원하는 연산을 통해 값을 앞으로 전달(feed-forward)하는 방법을 살펴보았습니다. 이제 이렇게 얻은 값을 우리가 원하는 값과의 차이를 계산하여 error를 뒤로 전달(back-propagation)하는 것을 해 보도록 하겠습니다.

예를 들어 우리가 원하는 값은 아래와 같이 **100**이라고 하였을 때, linear의 결과값 matrix의 합과 목표값과의 거리(error 또는 loss)를 구하고, 그 값에 대해서 **backward()**함수를 사용함으로써 gradient를 구합니다. 이때, error는 Variable class로 된 sclar로 표현 되어야 합니다. vector나 matrix의 형태여서는 안됩니다.

```python
objective = 100

x = torch.FloatTensor(16, 10)
x = Variable(x)
linear = MyLinear(10, 5)
y = linear(x)
loss = (objective - y.sum())**2

loss.backward()
```

위와 같이 구해진 각 parameter들의 gradient에 대해서 gradient descent 방법을 사용하여 error(loss)를 줄여나갈 수 있을 것 입니다.

## train\(\) and eval\(\)

```python
# Training...
linear.eval()
# Do some inference process.
linear.train()
# Restart training, again.
```

위와 같이 PyTorch는 **train()**과 **eval()** 함수를 제공하여 사용자가 필요에 따라 model에 대해서 훈련시와 추론시의 모드 전환을 쉽게 할 수 있도록 합니다. nn.Module을 상속받아 구현하고 생성한 객체는 기본적으로 training mode로 되어 있는데, **eval()**을 사용하여 module로 하여금 inference mode로 바꾸어주게 되면, (gradient를 계산하지 않도록 함으로써) inference 속도 뿐만 아니라, dropout 또는 batch-normalization과 같은 training과 inference 시에 다른 **forward()** 동작을 하는 module들에 대해서 각기 때에 따라 올바른 동작을 하도록 합니다. 다만, inference가 끝나면 다시 **train()**을 선언 해 주어, 원래의 훈련모드로 돌아가게 해 주어야 합니다.

## Example

이제까지 배운 것들을 활용하여 임의의 함수를 approximate하는 neural network를 구현 해 보도록 하겠습니다. 

1. Random으로 generate한 tensor들을 
1. 우리가 approximate하고자 하는 ground-truth 함수에 넣어 정답을 구하고, 
1. 그 정답($$y$$)과 neural network를 통과한 $$\hat{y}$$과의 차이(error)를 Mean Square Error(MSE)를 통해 구하여 
1. SGD를 통해서 optimize하도록 해 보겠습니다.

MSE의 수식은 아래와 같습니다.

$$
\begin{aligned}
&\mathcal{L}_{MSE}(x, y)=\frac{1}{N}\sum^N_{i=1}{(x_n - y_n)^2}
\end{aligned}
$$

먼저 1개의 linear layer를 가진 MyModel이라는 module을 선언합니다.

```python
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

class MyModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
                               
    def forward(self, x):
        y = self.linear(x)
                               
        return y
```

그리고 아래와 같이, 임의의 함수가 동작한다고 가정하겠습니다.

$$
\begin{aligned}
f(x_1, x_2, x_3) &= 3x_1 + x_2 - 2x_3
\end{aligned}
$$

해당 함수를 python으로 구현하면 아래와 같습니다. 물론 neural network 입장에서는 내부 동작 내용을 알 수 없는 함수 입니다.

```python
def ground_truth(x):
    return 3 * x[:, 0] + x[:, 1] - 2 * x[:, 2]
```

아래는 입력을 받아 feed-forward 시킨 후, back-propagation하여 gradient descent까지 하는 함수 입니다.

```python
def train(model, x, y, optim):
    # initialize gradients in all parameters in module.
    optim.zero_grad()
    
    # feed-forward
    y_hat = model(x)
    # get error between answer and inferenced.
    loss = ((y - y_hat)**2).sum() / x.size(0)
    
    # back-propagation
    loss.backward()
    
    # one-step of gradient descent
    optim.step()
    
    return loss.data[0]
```

그럼 위의 함수들을 사용 하기 위해서 hyper-parameter를 setting하겠습니다.

```python
batch_size = 1
n_epochs = 1000
n_iter = 10000

model = MyModel(3, 1)
optim = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.1)

print(model)
```

위의 setting 값을 사용하여 평균 loss 값이 **.001**보다 작을 때 까지 훈련 시킵니다.

```python
for epoch in range(n_epochs):
    avg_loss = 0
    
    for i in range(n_iter):
        x = Variable(torch.rand(batch_size, 3))
        y = Variable(ground_truth(x.data))

        loss = train(model, x, y, optim)
        
        avg_loss += loss
    avg_loss = avg_loss / n_iter

    # simple test sample to check the network.
    x_valid = Variable(torch.FloatTensor([[.3, .2, .1]]))
    y_valid = Variable(ground_truth(x_valid.data))

    model.eval()
    y_hat = model(x_valid)
    model.train()
    
    print(avg_loss, y_valid.data[0], y_hat.data[0, 0])  

    if avg_loss < .001: # finish the training if the loss is smaller than .001.
        break
```

위와 같이 임의의 함수에 대해서 실제로 neural network를 approximate하는 아주 간단한 예제를 살펴 보았습니다. 앞으로 책에서 다루어질 architecture들과 훈련 방법들도 이 예제의 연장선상에 지나지 않습니다.

## Using GPU

PyTorch는 당연히 GPU상에서 훈련하는 방법도 제공합니다. 아래와 같이 **cuda()**함수를 통해서 원하는 객체를 GPU memory상으로 copy(Variable 또는 Tensor의 경우)하거나 move(nn.Module의 하위 클래스인 경우) 시킬 수 있습니다.

```python
>>> # Note that tensor is declared in torch.cuda.
>>> x = torch.cuda.FloatTensor(16, 10)
>>> x = Variable(x)
>>> linear = MyLinear(10, 5)
>>> # .cuda() let module move to GPU memory.
>>> linear.cuda()
>>> y = linear(x)
```

또한, **cpu()**함수를 통해서 다시 PC의 memory로 copy하거나 move할 수 있습니다.