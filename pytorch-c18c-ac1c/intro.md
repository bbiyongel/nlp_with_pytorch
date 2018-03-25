# Deeplearning using PyTorch

## 장점

1. Dynamic Neural Networks
 - 기존의 static한 딥러닝 framework(TensorFlow, Theano)와 다르게 dynamic한 환경을 제공합니다.
1. Easy to Use
 - NumPy와 거의 유사한 문법과 개념을 공유합니다. 따라서 NumPy를 사용해 본 사용자라면 매우 쉽게 접하고 사용할 수 있습니다.

참고사이트: http://pytorch.org/about/

## 장비 구성

### CPU

잘 짜여진 PyTorch 코드는 대부분 GPU 사용량을 최대화 합니다. 따라서 보통의 경우 parallel한 연산은 모두 GPU에 넘어가게 되고, 일부 피할 수 없는 sequential한 연산만 남아 CPU의 사용량은 1개 core에 집중되어 100% 내외를 가리키게 됩니다. 따라서, Core의 숫자가 많은 것도 좋지만, 개별 코어의 clock이 높은 것이 더 중요합니다.

물론 본격적인 딥러닝을 하기 이전에 preprocessing 또는 word embedding의 단계에서는 CPU core가 많은 것이 좋을 수 있습니다. 따라서 이러한 작업간의 중요도를 잘 고려하여 CPU를 선택하면 됩니다. 잘 모르겠고 귀찮을 땐 그냥 i7-X700K를 하면 됩니다.

### RAM
### GPU
### Cooling System