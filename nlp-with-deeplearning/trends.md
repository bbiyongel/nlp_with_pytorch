# Recent Trends in NLP

## Conquering on Basic NLP

![](/assets/intro-rnnlm.png)

이전에 다루었던 대로, 인공지능의 다른 분야에 비해서 NLP는 가장 늦게 빛을 보기 시작하였다고 하였지만, 여러 task에 deep learning을 적용하려는 시도는 많이 이루어졌고, 진전은 있었습니다. 2010년에는 RNN을 활용하여 language modeling을 시도\[Mikolov et al.2010\]\[Sundermeyer at el.2012\]하여 기존의 n-gram 기반의 language model의 한계를 극복하려 하였습니다. 그리하여 기존의 n-gram 방식과의 interpolation을 통해서 더 나은 성능의 language model을 만들어낼 수 있었지만, 기존에 langauge model이 사용되던 음성인식과 기계번역에 적용되기에는 구조적인 한계\(Weighted Finite State Transeducer, WFST의 사용\)로 인해서 더 큰 성과를 거둘 수는 없었습니다. -- 애초에 n-gram 기반 언어모델의 한계는 WFST에 기반하였기 때문이라고도 볼 수 있습니다. 닭이 먼저냐, 달걀이 먼저냐의 문제와 같음.

![](/assets/intro-word2vec.png)

그러던 와중에 Mikolov는 2013년 Word2Vec\[Mikolov et al.2013\]을 발표합니다. 단순한 구조의 neural network를 사용하여 효과적으로 단어들을 hyper plane\(또는 vector space\)에 성공적으로 projection\(투영\) 시킴으로써, 본격적인 NLP 문제에 대한 딥러닝 활용의 신호탄을 쏘아 올렸습니다. 아래와 같이 우리는 고차원의 공간에 단어가 어떻게 배치되는지 알 수 있음으로 해서, deep learning을 활용하여 NLP에 대한 문제를 해결하고자 할 때에 network 내부는 어떤식으로 동작하는지에 대한 insight를 얻을 수 있었습니다.

![](/assets/intro-word-embedding.png)

이때까지는 문장이란 단어들의 time series이기 때문에, 당연히 Recurrent Neural Network\(RNN\)을 통해 해결해야 한다는 고정관념이 팽배해 있었습니다 -- Image=CNN, NLP=RNN. 하지만 2014년, Kim은 CNN만을 활용해 기존의 Text Classification보다 성능을 끌어올린 방법을 제시\[Kim et al.2014\]하며 한차례 파란을 일으킵니다. 이 방법은 word embedding vector와 결합하여 더 성능을 극대화 할 수 있었습니다. 위의 paper를 통해서 학계는 NLP에 대한 시각을 한차례 더 넓힐 수 있게 됩니다.

![](/assets/intro-cnn-text-classification.png)

이외에도 POS\(Part-of-Speech\) tagging, Sentence parsing, NER\(Named Entity Recognition\), SR\(Semantic Role\) labeling등에서도 기존의 state of the art를 뛰어넘는 성과를 이루냅니다. 하지만 딥러닝의 등장으로 인해 대부분의 task들이 end-to-end를 통해 문제를 해결하고자 함에따라, \(또한, 딥러닝 이전에도 이미 매우 좋은 성과를 내고 있었거나, 딥러닝의 적용 후에도 큰 성능의 차이가 없음에\) 큰 파란을 일으키지는 못합니다. -- 당연히 그정도는 좋아지는거 아니야? 이런 느낌...?

## Flourish of NLG

![](/assets/intro-word-alignment.png)

2014년 NLP에 큰 혁명이 다가옵니다. Sequence-to-Sequence의 발표\[Sutskever et al.2014\]에 이어, Attention 기법이 개발되어 성공적으로 기계번역에 적용\[Bahdanau et al.2014\]하여 큰 성과를 거둡니다.



## Breakthough with Attention, and Future

## Convergence with Reinforcement Learning



