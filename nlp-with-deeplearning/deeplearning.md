# Deep Learning

딥러닝의 시대가 오고 딥러닝은 하나하나 머신러닝의 분야들을 정복해 나가기 시작했습니다. 가장 먼저 두각을 나타낸 곳은 ImageNet이었지만, 가장 먼저 상용화 부문에서 빛을 본 것은 음성인식 분야였습니다. 음성인식은 여러 components 중에서 고작 하나인 GMM을 DNN으로 대체하였지만, 성능에 있어서 십수년의 정체를 뚫고 한 차례 큰 발전을 이루어냈습니다. 상대적으로 가장 나중에 빛을 본 곳은 NLP분야였습니다. 아마도 image classification과 음성인식의 phone recognition과 달리 NLP는 sequential한 데이터라는 것이 좀 더 장벽으로 다가왔으리라 생각됩니다. 하지만, 결국엔 attention의 등장으로 인해서 요원해 보이던 기계번역 분야마저 end-to-end deep learning에 의해서 정복되게 되었습니다.

## Brief Introduction to History of Deep Learning

### Before 2010's

인공신경망을 위시한 인공지능의 유행은 지금이 처음이 아닙니다. 이전까지 두 번의 대유행이 있었고, 그에 따른 두 번의 빙하기가 있었습니다. 80년대에 처음 back-propagation이 제안된 이후로, 모든 문제는 해결 된 듯 해 보였습니다. 하지만, 다시금 여러가지 한계점을 드러내며 침체기를 맞이하였습니다. 모두가 인공신경망의 가능성을 부인하던 2006년, Hinton 교수는 Deep Belief Networks을 통해 여러 층의 hidden layer를 효과적으로 pretraining 시킬 수 있는 방법을 제시하였습니다. 하지만, 아직까지 가시적인 성과가 나오지 않았기 때문에 모두의 관심을 집중 시킬 순 없었습니다. 아~ 그런가보다 하고 넘어가는 수준이었겠지요.

실제로 주변의 90년대의 빙하기를 겪어보신 세대 분들은 처음 딥러닝이 주목을 끌기 시작했을 때, 모두 부정적인 반응을 보이기 마련이었습니다. 계속 해서 최고 성능을 갈아치우며, 모두가 열광할 때에도, 단순한 잠깐의 유행일 것이라 생각하는 분들도 많았습니다. 하지만 점차 딥러닝은 여러 영역을 하나둘 정복 해 나가기 시작했습니다.

### Image Recognition

2012년 이미지넷에서 인공신경망을 이용한 AlexNet\(\[[Krizhevsky at el.2012](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf)\]\)은 경쟁자들을 큰 차이로 따돌리며 우승을 하고, 딥러닝의 시대의 서막을 올립니다. AlexNet은 여러 층의 Convolutional Layer을 쌓아서 architecture를 만들었고, 기존의 우승자들과 확연한 실력차를 보여주었습니다. 당시에 AlexNet은 3GB 메모리의 Nvidia GTX580을 2개 사용하여 훈련하였는데, 지금 생각하면 참으로 격세지감이 아닐 수 없습니다.

![Recent History of ImageNet](/assets/intro-imagenet.png)
Recent History of ImageNet

이후, ImageNet은 딥러닝의 경연장이 되었고, 거의 모든 참가자들이 딥러닝을 이용하여 알고리즘을 구현하였습니다. 결국, ResNet([[He et al.2015](https://arxiv.org/pdf/1512.03385.pdf)])은 Residual Connection을 활용하여 150층이 넘는 deep architecture를 구성하며 우승하였습니다.

### Speech Recognition

음성인식에 있어서도 딥러닝(당시에는 Deep Neural Network라는 이름으로 더욱 유명하였습니다.)을 활용하여 큰 발전을 이룩하였습니다. 오히려 이 분야에서는 vision분야에 비해서 딥러닝 기술을 활용하여 상용화에까지 성공한 더욱 인상적인 사례라고 할 수 있습니다.

![Traditional Speech Recognition System](https://www.esat.kuleuven.be/psi/spraak/demo/Recog/lvr_scheme.gif)
Traditional Speech Recognition System

사실 음성인식은 2000년대에 들어 큰 정체기를 맞이하고 있었습니다. GMM(Gaussian Mixture Model)을 통해 phone을 인식하고, 이를 HMM(Hidden Markov Model)을 통해 sequential 하게 modeling하여 만든 Acoustic Model (AM)과 n-gram기반의 Language Model (LM)을 WFST(Weighted Finite State Transeducer)방식을 통해 결합하는 전통적인 음성인식(Automatic Speech Recognition, ASR) 시스템은 위의 설명에서 볼 수 있듯이 너무나도 복잡한 구조와 함께 그 성능의 한계를 보이고 있었습니다.

![Accuracy of ASR](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAlTAAAAJDc0OTI3MzkyLTI2MTktNGE2Ni04MmI1LTJkODZhYjdlZWM1MQ.png)

그러던 중, 2012년 GMM을 DNN으로 대체하며, 십수년간의 정체를 단숨에 뛰어넘는 큰 혁명을 맞이하게 됩니다. (Vision, NLP에서 모두 보이는 익숙한 패턴입니다.) 그리고 점차 AM전체를 LSTM으로 대체하고, 또한 end-to-end model([[Chiu et al.2017](https://arxiv.org/pdf/1712.01769.pdf)])이 점점 저변을 넓혀가고 있는 추세입니다.

### Machine Translation

물밀듯이 밀려오는 딥러닝의 침략 앞에서 기계번역 또한 예외일 순 없었습니다. 딥러닝 이전의 기계번역은 통계 기반 기계번역(Statistical Machine Translation, SMT)가 지배하고 있었습니다. 비록 SMT는 규칙기반의 번역방식(Rule based Machine Translation, RBMT)에 비해서 언어간 확장이 용이한 장점이 있었고, 성능도 더 뛰어났지만, 음성인식과 마찬가지로 SMT는 역시 너무나도 복잡한 구조를 지니고 있었습니다. 

![](http://www.kecl.ntt.co.jp/rps/_src/sc1134/innovative_3_1e.jpg)

2014년 Sequence-to-sequence(seq2seq)라는 architecture가 소개 되며, end-to-end neural machine translation의 시대가 열리게 되었습니다. 

![History of Machine Translation](http://iconictranslation.com/wp-content/uploads/2017/06/NMT-Graph-2-a.png)

Seq2seq를 기반으로 attention mechanism([[Bahdanau et al.2014](https://arxiv.org/pdf/1409.0473.pdf)], [[Luong et al.2015](https://arxiv.org/pdf/1508.04025.pdf)])이 제안되며 결국 기계번역은 Neural Machine Translation에 의해서 대통합이 이루어지게 됩니다.

참고 사이트: 
https://devblogs.nvidia.com/introduction-neural-machine-translation-with-gpus/
https://devblogs.nvidia.com/introduction-neural-machine-translation-gpus-part-2/
https://devblogs.nvidia.com/introduction-neural-machine-translation-gpus-part-3/

### Generative Learning

Neural Network은 pattern classification에 있어서 타 알고리즘에 비해서 너무나도 압도적인 성능을 보여주었기 때문에, image recognition, text classification과 같은 단순한 분류 문제(classification or discriminative learning)는 금방 정복되고 더 이상 연구자들의 흥미를 끌 수 없었습니다.

> 각 방식이 흥미를 두고 있는 것:
- Discriminative learning
$$
\hat{Y} = argmax_{Y}P(Y|X).
$$
- Generative learning
$$
P(X), itself.
$$

따라서, 곧 연구자들은 또 다른 흥미거리를 찾아 나섰는데, 그것은 Generative Learning이었습니다. 기존의 classification 문제는 $$ X $$가 주어졌을 때, 알맞은 $$ Y $$를 찾아내는 것에 집중했다면, 이제는 $$ X $$ 자체에 집중하기 시작한 것 입니다. 예를 들어 기존에는 사람의 얼굴 사진이 주어지면 남자인지 여자인지, 또는 더 나아가 이 사람이 누구인지 알아내는 것이었다면, 이제는 얼굴 자체를 묘사할 수 있는 모델을 훈련하고자 하였습니다.

![Modeling face based on age](http://www.i-programmer.info/images/stories/News/2017/feb/A/age.jpg)

이러한 과정에서 Adversarial learning (GAN)이나 Variational Auto-encoder (VAE)등이 주목받게 되었습니다.

## Paradigm Shift on NLP from Traditional to Deep Learning

![](/assets/intro-paradigm-shift.png)![](/assets/intro-traditional-nlp.png)

### Word2Vec

![](/assets/intro-word-embedding.png)![](/assets/intro-word2vec.png)![](/assets/intro-nlp-symbolic-vs-neural.png)

### LSTM and GRU

### Attention

### Reinforcement Learning

## Conclusion

![](/assets/intro-end-2-end-nlp-deep-learning.png)

