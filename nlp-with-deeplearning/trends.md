# Recent Trends in NLP

## Conquering on Basic NLP

![](/assets/intro-rnnlm.png)

이전에 다루었던 대로, 인공지능의 다른 분야에 비해서 NLP는 가장 늦게 빛을 보기 시작하였다고 하였지만, 여러 task에 deep learning을 적용하려는 시도는 많이 이루어졌고, 진전은 있었습니다. 2010년에는 RNN을 활용하여 language modeling을 시도\[Mikolov et al.2010\]\[Sundermeyer at el.2012\]하여 기존의 n-gram 기반의 language model의 한계를 극복하려 하였습니다. 그리하여 기존의 n-gram 방식과의 interpolation을 통해서 더 나은 성능의 language model을 만들어낼 수 있었지만, 기존에 langauge model이 사용되던 음성인식과 기계번역에 적용되기에는 구조적인 한계\(Weighted Finite State Transeducer, WFST의 사용\)로 인해서 더 큰 성과를 거둘 수는 없었습니다. -- 애초에 n-gram 기반 언어모델의 한계는 WFST에 기반하였기 때문이라고도 볼 수 있습니다. 닭이 먼저냐, 달걀이 먼저냐의 문제와 같음.

![](/assets/intro-word2vec.png)

그러던 와중에 Mikolov는 2013년 Word2Vec\[Mikolov et al.2013\]을 발표합니다. 단순한 구조의 neural network를 사용하여 효과적으로 단어들을 hyper plane\(또는 vector space\)에 성공적으로 projection\(투영\) 시킴으로써, 본격적인 NLP 문제에 대한 딥러닝 활용의 신호탄을 쏘아 올렸습니다. 아래와 같이 우리는 고차원의 공간에 단어가 어떻게 배치되는지 알 수 있음으로 해서, deep learning을 활용하여 NLP에 대한 문제를 해결하고자 할 때에 network 내부는 어떤식으로 동작하는지에 대한 insight를 얻을 수 있었습니다.

![](/assets/intro-word-embedding.png)

이때까지는 문장이란 단어들의 time series이기 때문에, 당연히 Recurrent Neural Network(RNN)을 통해 해결해야 한다는 고정관념이 팽배해 있었습니다 -- Image=CNN, NLP=RNN. 하지만 2014년, Kim은 CNN만을 활용해 기존의 Text Classification보다 성능을 끌어올린 방법을 제시[Kim et al.2014]하며 한차례 파란을 일으킵니다. 



## Flourish of NLG

## Breakthough with Attention, and Future

## Convergence with Reinforcement Learning



