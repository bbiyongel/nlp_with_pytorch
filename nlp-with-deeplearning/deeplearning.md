# Deep Learning

딥러닝의 시대가 오고 딥러닝은 하나하나 머신러닝의 분야들을 정복해 나가기 시작했습니다. 가장 먼저 두각을 나타낸 곳은 ImageNet이었지만, 가장 먼저 상용화 부문에서 빛을 본 것은 음성인식 분야였습니다. 음성인식은 여러 components 중에서 고작 하나인 GMM을 DNN으로 대체하였지만, 성능에 있어서 십수년의 정체를 뚫고 한 차례 큰 발전을 이루어냈습니다. 상대적으로 가장 나중에 빛을 본 곳은 NLP분야였습니다. 아마도 image classification과 음성인식의 phone recognition과 달리 NLP는 sequential한 데이터라는 것이 좀 더 장벽으로 다가왔으리라 생각됩니다. 하지만, 결국엔 attention의 등장으로 인해서 요원해 보이던 기계번역 분야마저 end-to-end deep learning에 의해서 정복되게 되었습니다.

## Brief Introduction to History of Deep Learning

### Before 2010's

인공신경망을 위시한 인공지능의 유행은 지금이 처음이 아닙니다. 이전까지 두 번의 대유행이 있었고, 그에 따른 두 번의 빙하기가 있었습니다. 80년대에 처음 back-propagation이 제안된 이후로, 모든 문제는 해결 된 듯 해 보였습니다. 하지만, 다시금 여러가지 한계점을 드러내며 침체기를 맞이하였습니다. 모두가 인공신경망의 가능성을 부인하던 2006년, Hinton 교수는 Deep Belief Networks을 통해 여러 층의 hidden layer를 효과적으로 pretraining 시킬 수 있는 방법을 제시하였습니다. 하지만, 아직까지 가시적인 성과가 나오지 않았기 때문에 모두의 관심을 집중 시킬 순 없었습니다. 아~ 그런가보다 하고 넘어가는 수준이었겠지요.

실제로 주변의 90년대의 빙하기를 겪어보신 세대 분들은 처음 딥러닝이 주목을 끌기 시작했을 때, 모두 부정적인 반응을 보이기 마련이었습니다. 계속 해서 최고 성능을 갈아치우며, 모두가 열광할 때에도, 단순한 잠깐의 유행일 것이라 생각하는 분들도 많았습니다. 하지만 점차 딥러닝은 여러 영역을 하나둘 정복 해 나가기 시작했습니다.

### Computer Vision

2012년 이미지넷에서 인공신경망을 이용한 AlexNet\(\[[Krizhevsky at el.2012](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf)\]\)은 경쟁자들을 큰 차이로 따돌리며 우승을 하고, 딥러닝의 시대의 서막을 올립니다. AlexNet은 여러 층의 Convolutional Layer을 쌓아서 architecture를 만들었고, 기존의 우승자들과 확연한 실력차를 보여주었습니다. 당시에 AlexNet은 3GB 메모리의 Nvidia GTX580을 2개 사용하여 훈련하였는데, 지금 생각하면 참으로 격세지감이 아닐 수 없습니다.

![](/assets/intro-imagenet.png)

이후, ImageNet은 딥러닝의 경연장이 되었고, 거의 모든 참가자들이 딥러닝을 이용하여 알고리즘을 구현하였습니다. 결국, ResNet([[He at el.2015](https://arxiv.org/pdf/1512.03385.pdf)])은 Residual Connection을 활용하여 150층이 넘는 deep architecture를 구성하며 우승하였습니다.

### Speech Recognition

음성인식에 있어서도 딥러닝(당시에는 Deep Neural Network라는 이름으로 더욱 유명하였습니다.)을 활용하여 큰 발전을 이룩하였습니다. 오히려 이 분야에서는 vision분야에 비해서 딥러닝 기술을 활용하여 상용화에까지 성공한 더욱 인상적인 사례라고 할 수 있습니다.

사실 음성인식은 2000년대에 들어 큰 정체기를 맞이하고 있었습니다. GMM(Gaussian Mixture Model)을 통해 phone을 인식하고, 이를 HMM(Hidden Markov Model)을 통해 sequential 하게 modeling하여 만든 Acoustic Model (AM)과 n-gram기반의 Language Model (LM)을 WFST(Weighted Finite State Transeducer)방식을 통해 결합하는 전통적인 

## Paradigm Shift on NLP from Traditional to Deep Learning

![](/assets/intro-paradigm-shift.png)![](/assets/intro-traditional-nlp.png)

### Word2Vec

![](/assets/intro-word-embedding.png)![](/assets/intro-word2vec.png)![](/assets/intro-nlp-symbolic-vs-neural.png)

### LSTM and GRU

### Attention

### Reinforcement Learning

## Conclusion

![](/assets/intro-end-2-end-nlp-deep-learning.png)

