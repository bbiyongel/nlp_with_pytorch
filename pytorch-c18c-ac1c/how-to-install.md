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

Python을 처음 접하는 많은 사용자들이 2.7과 3.6 사이에서 어떤 것을 택해야 할 지 고민하게 됩니다. 처음 python을 접하는 사람들은 3.6을 택하는 것을 추천합니다. 특히, 이 책에서 다루는 NLP와 관련된 text encoding의 default가 UTF-8로 되어 있어서 훨씬 더 편리하게 사용할 수 있습니다. 다만, 2.7에서 작성된 코드를 3.6에서 사용하기 위해서는 코드를 약간 수정해야 할 필요성이 있습니다. \(대부분의 경우 3.6에서 작성한 코드는 2.7에서 잘 돌아갈 가능성이 훨씬 더 높습니다.\)

## PyTorch

![https://twitter.com/karpathy/status/868178954032513024](/assets/pytorch-intro-Karpathy.png)
[Image from [Karpathy's twitter](https://twitter.com/karpathy/status/868178954032513024)]