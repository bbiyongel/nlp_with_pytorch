# How to Install

Linux \(Ubuntu\) 기준으로 PyTorch를 설치하고 실행하는 것을 살펴 보도록 하겠습니다.

## Anaconda

대부분 linux를 설치하면 기본적으로 python이 설치되어 있는 것을 볼 수 있습니다. 대부분의 경우 아래와 같은 경로에 설치되어 있습니다.

```bash
$ sudo which python
/usr/bin/python
```

이 경우에는 시스템 전체 사용자들이 공통으로 사용하는 python이기 때문에 anaconda를 설치하여 해당 사용자만을 위한 python을 설치하고, 그 위에 여러 package를 자유롭게 install 또는 uninstall 하는 것이 편리합니다. 또한, 경우에 따라 발생할 수 있는 권한 문제에서도 훨씬 자유롭습니다. 따라서 Anaconda를 사용하는 것을 권장합니다. Anaconda는 아래의 주소에서 다운로드 받을 수 있습니다.

> [https://www.anaconda.com/download/\#linux](https://www.anaconda.com/download/#linux)

또한 많은 package가 기본으로 설치되는 anaconda와 달리 Miniconda를 설치하여 훨씬 더 가볍게 사용할 수도 있습니다.

### 2.7 vs 3.6

Python을 처음 접하는 많은 사용자들이 2.7과 3.6 사이에서 어떤 것을 택해야 할 지 고민하게 됩니다. 처음 python을 접하는 사람들은 3.6을 택하는 것을 추천합니다. 특히, 이 책에서 다루는 NLP와 관련된 text encoding의 default가 UTF-8로 되어 있어서 훨씬 더 편리하게 사용할 수 있습니다. 따라서, 시간을 아끼기 위해서는 3.6으로 시작할 것을 권장합니다. 다만, 2.7에서 작성된 코드를 3.6에서 사용하기 위해서는 코드를 약간 수정해야 할 필요성이 있습니다. \(참고로, 대부분의 경우 3.6에서 작성한 코드는 2.7에서 잘 돌아갈 가능성이 훨씬 더 높습니다.\)

## 왜 PyTorch 인가?

![](/assets/pytorch-intro-logo.png)

Tensorflow를 개발한 Google에 맞서, PyTorch는 Facebook의 주도하에 개발이 진행되고 있습니다. 자체 딥러닝 전용 H/W인 TPU를 가지고 있어 상대적으로 Nvidia GPU에서 보다 자유로운 Google과 달리, PyTorch는 Nvidia도 참여한 project이기 때문에 Nvidia의 CUDA GPU에 더욱 최적화 되어 있습니다. 실제로도, Nvidia에서도 적극 PyTorch를 권장하는 모습이며, 특히 NLP 분야에서는 Tensorflow에 비하여 적극 권장하기도 합니다.

![](/assets/pytorch-intro-company.png)

일찌감치 Tensorflow를 내세운 Google과 달리, PyTorch는 그에비해 훨씬 뒤늦게 deep learning framework 개발에 뛰어들었기 때문에, 상대적으로 훨씬 적은 유저풀을 갖고 있습니다.

![](https://cdn-images-1.medium.com/max/2000/1*8a2Nz2SnCgT9UFl7rSaywg.png)

하지만, PyTorch가 가진 장점과 매력 때문에, 산업계보다는 학계에서 적극적으로 PyTorch의 사용을 늘려가고 있는 추세이며, 이러한 트렌드는 산업계에도 점점 퍼져나가고 있습니다. 따라서, Tensorflow는 paper를 구현한 수많은 github source code와 pretrain된 model parameter가 있는 것이 장점이긴 하지만, PyTorch도 빠르게 따라잡고 있는 추세 입니다. -- 하지만 아직은 Tensorflow의 아성을 넘기에는 부족합니다.

![https://twitter.com/karpathy/status/868178954032513024](/assets/pytorch-intro-Karpathy.png)  
\[Image from [Karpathy's twitter](https://twitter.com/karpathy/status/868178954032513024)\]

Tesla의 AI 수장인 Karpathy는 자신의 트위터에서 파이토치를 찬양하였습니다. 그럼 무엇이 그를 찬양하도록 만들었는지 좀 더 알아보도록 하겠습니다. PyTorch는 major deep learning framework 중에서 가장 늦게 나온 편인 만큼, 그동안 여러 framework의 장점을 모두 갖고 있습니다.

* Python First, 깔끔한 코드
  * 먼저 Tensorflow와 달리 Python First를 표방한 PyTorch는 tensor연산과 같이 속도에 크리티컬 한 부분을 제외하고는 대부분의 모듈이 python으로 짜여 있습니다. 따라서 코드가 깔끔합니다.
* NumPy/SciPy과 뛰어난 호환성
  * Theano의 장점인 NumPy와의 호환성이 PyTorch에도 그대로 들어왔습니다. 따라서 기존 numpy를 사용하던 사용자들은 처음 파이토치를 접하더라도 큰 위화감 없이 그대로 적응할 수 있습니다.
* Autograd
  * 단지 값을 앞으로 전달\(feed-forward\)시키며 차례차례 계산 한 것일 뿐인데, **backward\(\)** 호출 한번에 gradient를 구할 수 있습니다.
* Dynamic Graph
  * Tensorflow의 경우 session이라는 개념이 있어서, session이 시작되면 model architecture등의 graph 구조의 수정이 어려웠습니다. 하지만, PyTorch는 그러한 개념이 없어 편리하게 사용 할 수 있습니다.



