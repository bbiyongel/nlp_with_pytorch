# Applications

왜 Language Modeling이 중요할까요? 다양한 분야에서 널리 사용되기 때문입니다. 대표적인 활용 분야는 아래와 같습니다.

## Automatic Speech Recognition (ASR)

음성인식 시스템을 구성할 때, 언어모델은 중요하게 쓰입니다. 사실 실제 사람의 경우에도 말을 들을 때 언어모델이 굉장히 중요하게 작용합니다. 어떤 단어를 발음하고 있는지 명확하게 알아듣지 못하더라도, 머릿속에 저장되어 있는 언어모델을 이용하여 알아듣기 때문입니다. 예를 들어, 우리는 갑자기 쌩뚱맞은 주제로 대화를 전환하게 되면 보통 한번에 잘 못알아듣는 경우가 많습니다. Computer의 경우에도 음소별 classification의 성능은 이미 사람보다 뛰어납니다. 다만, 사람에 비해 context 정보를 활용할 수 있는 능력, ***눈치***가 없기 때문에 음성인식률이 떨어지는 경우가 많습니다. 따라서, 그나마 좋은 language model을 정의하여 사용함으로써, 음성인식의 정확도를 높일 수 있습니다.

![Traditional Speech Recognition System](https://www.esat.kuleuven.be/psi/spraak/demo/Recog/lvr_scheme.gif)<br>
(Traditional ASR System based on WFST, Image from web)

Language Model이 가장 중요한 역할을 차지하는 음성인식을 대표 삼아, Language Model이 다른 모델들과 결합하여 어떻게 쓰이는지 알아보도록 하겠습니다. 아래는 Speech Recognition 시스템의 수식을 개략적으로 나타낸 것 입니다. 음성 신호 $$ X $$가 주어졌을 때 확률을 최대로 하는 문장 $$ \hat{Y} $$를 구하는 것이 목표 입니다.

$$
\begin{aligned}
\hat{Y} = argmax P(Y|X) &= argmax\frac{P(X|Y)P(Y)}{P(X)} \\
\text{where X is audio signal and }&\text{Y is word sequence, }Y = \{ y_1, y_2, \dots y_n \}.
\end{aligned}
$$

그럼 Bayes Theorem에 의해서 수식은 AM과 LM 부분으로 나뉠 수 있습니다. 그리고 밑변의 prior $$ P(X) $$는 날려버릴 수 있습니다.

$$
argmax\frac{P(X|Y)P(Y)}{P(X)} = argmax \underbrace{P(X|Y)}_{\text{AM}}\underbrace{P(Y)}_{\text{LM}}
$$

여기서 $$ P(X|Y) $$가 Acoustic Model (AM)을 의미하고, $$ P(Y) $$는 Language Model (LM)을 의미 합니다. 즉, $$ P(Y) $$는 문장의 확률을 의미하고, $$ P(X|Y) $$는 문장(= sequence of word = sequence of phone)이 주어졌을 때, audio signal이 나타날 확률을 나타냅니다.

이처럼 LM은 AM과 결합하여, context에 기반하여 다음 단어를 예측하도록 함으로써, 음성인식의 성능을 더울 향상 시킬 수 있습니다.

## Machine Translation (MT)

번역 시스템을 구성 할 때에도 언어모델은 중요한 역할을 합니다. 기존 SMT 시스템에서는 explicit하게 LM이 translation model과 결합하였지만, Neural Machine Translation (NMT)의 경우에도 LM이 마찬가지로 매우 중요한 역할을 차지 합니다. 예를 들어, NMT에서는 source sentence를 분석하여 의미를 파악한 후 target sentence를 만들어 낼 때에, 언어 모델을 기반으로 문장을 다시 생성 해 냅니다. 더 자세한 내용은 다음 챕터에서 다루도록 하겠습니다.

![](http://www.kecl.ntt.co.jp/rps/_src/sc1134/innovative_3_1e.jpg)<br>
(Statistical Machine Translation, SMT, system, Image from web)

## Optical Character Recognition (OCR)

광학문자인식 시스템을 만들 때에도 언어모델이 사용 됩니다. 사진에서 추출하여 글자를 인식 할 때에 언어모델의 확률의 도움을 받아 글자나 글씨를 인식합니다.

![](https://doi.ieeecomputersociety.org/cms/Computer.org/dl/trans/tp/2013/10/figures/ttp20131024131.gif)<br>
(OCR system, Image from web)

## Natural Language Generation

사실 위에 나열한 ASR, MT, OCR도 주어진 정보를 바탕으로 문장을 생성해내는 NLG에 속한다고 볼 수 있습니다. 이외에도 NLG가 적용 될 수 있는 영역은 굉장히 많습니다. 주어진 정보를 바탕으로 뉴스 기사를 쓸 수도 있고, 주어진 뉴스 기사를 요약하여 제목을 생성 해 낼 수도 있습니다. 또한, 사용자의 응답에 따라 대답을 생성 해 내는 chatbot도 생각 해 볼 수 있습니다.

## Others

이외에도 여러가지 영역에 정말 다양하게 사용됩니다. 검색엔진에서 사용자가 검색어를 입력하는 도중에 밑에 drop-down으로 제시되는 검색어 완성 등에도 language model이 사용 될 수 있습니다.

## Why Language Model is important?

사실 LM 자체의 단독 활용도는 실제 필드에서 그렇게 크지 않습니다. 하지만, Natural Language Generation(NLG)의 가장 기본이라고 할 수 있습니다. NLG는 현재 딥러닝을 활용한 NLP 분야에서 가장 큰 연구주제 입니다. 기계번역에서부터 챗봇까지 모두 NLG의 영역에 포함된다고 할 수 있습니다. 이러한 NLG의 기본 초석이 되는 것이 바로 언어모델(LM) 입니다. 따라서, LM 자체의 활용도는 그 중요성에 비해 떨어질지언정, LM의 중요성과 그 역할은 부인할 수 없습니다.