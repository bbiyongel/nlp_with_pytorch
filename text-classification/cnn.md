# CNN을 통한 접근 방법

이번 섹션에서는 Convolutional Nueral Network (CNN) Layer를 활용한 텍스트 분류에 대해 다루어 보겠습니다. CNN을 활용한 방법은 [[Kim at el.2014]](https://arxiv.org/pdf/1408.5882.pdf)에 의해서 처음 제안되었습니다. 사실 이전까지 딥러닝을 활용한 자연어처리는 Recurrent Nueral Networ (RNN)에 국한되어 있는 느낌이 매우 강했습니다. 텍스트 문장은 여러 단어로 이루어져 있고, 그 문장의 길이가 문장마다 상이하며, 문장 내의 단어들은 같은 문장 내의 단어에 따라서 영향을 받기 때문입니다. 

좀 더 비약적으로 표현하면 $t$ time-step에 등장하는 단어 $w_t$는 이전 time-step에 등장한 단어들 $w_1,\cdots,w_{t_1}$에 의존하기 때문입니다. (물론 실제로는 $t$ 이후에 등장하는 단어들로부터도 영향을 받습니다.) 따라서 시간 개념이 도입되어야 하기 때문에, RNN의 사용은 불가피하다고 생각되었습니다. 하지만 앞서 소개한 [[Kim at el.2014]](https://arxiv.org/pdf/1408.5882.pdf) 논문에 의해서 새로운 시각이 열리게 됩니다.

## Convolution Operation

사실 널리 알려졌다시피, CNN은 영상처리(or Computer Vision) 분야에서 매우 큰 성과를 거두고 있었습니다. CNN의 동기 자체가, 기존의 전통적인 영상처리에서 사용되던 각종 convolution 필터(filter or kernel)를 자동으로 학습하기 위함이기 때문입니다.

### Convolution Filter

전통적인 영상처리 분야에서는 손으로 한땀한땀 만들어낸 필터를 사용하여 윤곽선을 검출하는 등의 전처리 과정을 거쳐, 얻어낸 피쳐(feature)들을 통해 객체 탐지(object detection)등을 구현하곤 하였습니다. 예를 들어 주어진 이미지에서 윤곽선(edge)을 찾기 위한 convolution 필터는 아래와 같습니다.

![Sobel Filters for vertial and horizontal edges](/assets/tc-cnn-sobel-filter.gif)

이 필터를 이미지에 적용하면 아래와 같은 결과를 얻을 수 있습니다.

![An image before Sobel filter (from Wikipedia)](https://upload.wikimedia.org/wikipedia/commons/f/f0/Valve_original_%281%29.PNG)

![Image after applying Sobel filter (from Wikipedia)](https://upload.wikimedia.org/wikipedia/commons/d/d4/Valve_sobel_%283%29.PNG)

이처럼 전처리 서브모듈에서 여러 필터들을 문제에 따라 적용하여 피쳐들을 얻어낸 이후에, 다음 서브모듈을 적용하여 주어진 문제를 해결하는 방식이었습니다.

## Convolutional Neural Network Layer

만약 문제에 따라서 필요한 convoltuion 필터를 자동으로 찾아준다면 어떻게 될까요? CNN이 바로 그러한 역할을 해주게 됩니다. Convolution 연산을 통해 feed-forward 된 값에 back-propagation을 하여, 더 나은 convolution 필터 값을 찾아나가게 됩니다. 따라서 마지막에 loss 값이 수렴 한 이후에는, 해당 문제에 딱 맞는 여러 종류의 convolution 필터를 찾아낼 수 있게 되는 것 입니다.

![Convolution 연산을 적용하는 과정](/assets/tc-convolution.png)

$$
\begin{aligned}
y_{1,1}&=x_{1,1}*k_{1,1}+\cdots+x_{3,3}*k_{3,3} \\
&=\sum_{i=1}^3{\sum_{j=1}^3{x_{i,j}*k_{i,j}}}
\end{aligned}
$$

Convolution 필터 연산의 forward는 위와 같습니다. 필터(또는 커널)가 주어진 이미지 위에서 차례대로 convolution 연산을 수행합니다. 보다시피, 상당히 많은 연산이 병렬(parallel)로 수행될 수 있음을 알 수 있습니다.

기본적으로는 convolution 연산의 결과물은 필터의 크기에 따라 입력에 비해서 크기가 줄어듭니다. 위의 그림에서도 필터의 크기가 $3\times3$ 이므로, $6\times6$ 입력에 적용하면 $4\times4$ 크기의 결과물을 얻을 수 있습니다. 따라서 입력과 같은 크기를 유지하기 위해서는 결과물의 바깥에 패딩(padding)을 추가하여 크기를 유지할 수도 있습니다.

## How to use CNN to NLP

[article from WildML](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

## 구조 설계

## 설명

## 코드