# Natural Language Processing with PyTorch

## PyTorch를 활용한 자연어처리

이 책은 딥러닝을 활용하여 자연어처리(Natural Language Processing)를 실무에 적용하고자 하는 분들을 대상으로 쓰여진 책 입니다. 사실 자연어처리는 그 필요성에 비하여 인공지능의 다른 분야(영상처리 등)에 비해 시중에 나와있는 도서나 인터넷 상의 정보가 매우 부족한 것이 현실입니다. 특히, Sequence-to-Sequence를 활용한 자연어 생성 분야는 그 활용 가능성이 비해 자료가 전무하다시피 합니다. 따라서 자연어처리를 공부하고자 하는 분들이 많은 어려움을 겪고 있습니다. 이런 어려움을 해소하고자, 여러해에 걸친 패스트 캠퍼스의 강의 경험을 담아, 독자들이 기본기부터 제대로 이해할 수 있도록 책의 내용을 구성하였습니다. 따라서 이 책은 입문자들을 위한 초급의 내용부터 실무자들을 위한 심화의 내용까지 모두 아우를 수 있도록 많은 내용을 담고자 하였고, 실무에 바로 투입이 가능한 정도의 파이토치 실습 코드를 활용하여 독자들의 이해를 돕고자 합니다.

실제 시스템을 구현하며 얻은 경험과 인사이트들을 담았고, 배경이 되는 수학적인 이론에서부터, 실제 PyTorch 코드, 그리고 실전에서의 꼭 필요한 직관적인 개념들을 담을 수 있도록 하였습니다. 그러므로, 현재 딥러닝을 활용한 최신 기술 뿐만 아니라, 딥러닝 이전의 기존의 전통적인 방식부터 차근차근 설명하여, 왜 이 기술이 필요하고, 어떻게 발전 해 왔으며, 어떤 부분이 성능 개선을 만들어냈는지 쉽게 이해할 수 있도록 설명하고자 합니다.

수식이 낯설은 입문자 분들은 수식은 좀 더 과감히 건너뛰고 읽으셔도 좋습니다. 먼저 한번 읽은 후에 큰 그림을 이해하고 다시 읽으며 수식을 이해하면 좀 더 좋을 것 같습니다. 좀 더 깊은 내용을 알고 싶은 독자분들은 수식까지 모두 집중하여 읽어주길 권장 합니다.

* Github Repo: [https://github.com/kh-kim/nlp\_with\_pytorch](https://github.com/kh-kim/nlp_with_pytorch)
* Gitbook: [https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/)

### Pre-requisites

자연어처리 분야에 좀 더 집중하기 위하여, 시중의 도서나 인터넷에서 쉽게 접할 수 있는, 머신러닝과 딥러닝을 입문하기 위한 내용(예를 들어 역전파 알고리즘)에 대한 설명은 최소화 하였습니다. 따라서 이 책의 대부분의 내용은 독자가 아래의 내용에 대해서 지식이 있거나 다른 자료를 통해 같이 공부하며 읽을 것을 권장 합니다.

* Python
* Calculus, Linear Algebra
* Probability and Statistics
* Basic Machine Learning
  * Objective / loss function
  * Linear / logistic regression
  * Gradient descent
* Basic Deep Learning
  * Back-propagation
  * Activation function

## 지은이: 김기현(Kim, Ki Hyun)

![김기현](../assets/me.jpeg){ width=300px }

### 연락처

|Name|Kim, Ki Hyun|
|-|-|
|email|nlp.with.deep.learning@gmail.com|
|linkedin|[https://www.linkedin.com/in/ki-hyun-kim/](https://www.linkedin.com/in/ki-hyun-kim/)|
|github:|[https://github.com/kh-kim/](https://github.com/kh-kim/)|

### 약력

* Principal Machine Learning Engineer @ [Makinarocks](http://makinarocks.ai)
* Machine Learning Researcher @ SKPlanet 
  * Neural Machine Translation: [Global 11번가](http://global.11st.co.kr/html/en/main_en.html?trlang=en)
* Machine Learning Engineer @ Ticketmonster 
  * Recommender System: [TMON](http://www.ticketmonster.co.kr/)
* Researcher @ ETRI 
  * Automatic Speech Translation: GenieTalk \[[Android](https://play.google.com/store/apps/details?id=com.hancom.interfree.genietalk&hl=ko)\], \[[iOS](https://itunes.apple.com/kr/app/지니톡-genietalk/id1104930501?mt=8)\], \[[TV Ads](https://www.youtube.com/watch?v=Jda0G0yhWpM)\]

## 패스트캠퍼스 강의 소개
![자연어처리 심화 캠프](../assets/fastcampus_lecture.png)

현재 이 책을 바탕으로 패스트캠퍼스에서 [자연어처리 입문 캠프](https://www.fastcampus.co.kr/data_camp_nlpbasic/), [자연어처리 심화 캠프](http://www.fastcampus.co.kr/data_camp_nlpadv/)도 진행하고 있습니다.

이 저작물은 크리에이티브 커먼즈 [저작자표시-비영리-동일조건변경허락(BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/2.0/kr/)에 따라 이용할 수 있습니다.

![저작자표시-비영리-동일조건변경허락(BY-NC-SA)](../assets/ccl.png)
