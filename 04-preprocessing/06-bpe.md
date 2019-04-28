# 서브워드 단위 분절 (Byte Pair Encoding, BPE)

Byte Pair Encoding (BPE) 알고리즘을 통한 서브워드(sub-word) 단위 분절은 [[Sennrich at el.2015]](https://arxiv.org/pdf/1508.07909.pdf)에서 처음 제안되었고, 현재는 주류 전처리 방법으로 자리잡아 사용되고 있습니다. 서브워드 단위 분절 기법은 기본적으로 단어는 의미를 가진 더 작은 서브워드들의 조합으로 이루어진다는 가정 하에 적용 되는 알고리즘입니다. 실제로 영어나 한국어의 경우에도 라틴어와 한자를 기반으로 형성 된 언어이기 때문에, 많은 단어들이 서브워드들로 구성되어 있습니다. 따라서 적절한 서브워드의 발견을 통해서 서브워드 단위로 쪼개어 주면 어휘수를 줄일 수 있고 희소성(sparsity)을 효과적으로 감소시킬 수 있습니다.

|언어|단어|조합|
|-|-|-|
|영어|Concentrate|con(=together) + centr(=center) + ate(=make)|
|한국어|집중(集中)|集(모을 집) + 中(가운데 중)|

희소성 감소 이외에도, 가장 대표적인 서브워드 단위 분절로 얻는 효과는 언노운(unknown, UNK) 토큰에 대한 효율적인 대처 입니다. 자연어 생성(NLG)을 포함한 대부분의 딥러닝 자연어처리 알고리즘들은 문장을 입력으로 받을 때 단순히 단어들의 시퀀스로써 받아들이게 됩니다. 따라서 UNK가 나타나게 되면 이후의 언어모델의 확률은 매우 망가져버리게 됩니다. 따라서 적절한 문장의 임베딩(인코딩)또는 생성이 어렵습니다. <comment> 특히 문장 생성의 경우에는 이전 단어를 기반으로 다음 단어를 예측하기 때문에 더더욱 어렵습니다. </comment>

하지만 서브워드 단위 분절을 통해서 신조어나 오타(typo) 같은 UNK에 대해서 서브워드 단위나 캐릭터(character) 단위로 쪼개줌으로써 기존에 훈련 데이터에서 보았던 토큰들의 조합으로 만들어버릴 수 있습니다. 즉, UNK 자체를 없앰으로서 효율적으로 UNK에 대처할 수 있고, 자연어처리 알고리즘의 결과물 품질을 향상시킬 수 있습니다. 다만, 한글과 영어만을 가지고 훈련한 알고리즘 또는 모델에 '아랍어'와 같이 전혀 보지 못한 캐릭터가 등장한다면 당연히 UNK로 치환될 것 입니다.

## 예제

아래와 같이 BPE를 적용하면 원래의 띄어쓰기 공백 이외에도 BPE의 적용으로 인한 공백이 추가됩니다. 따라서 원래의 띄어쓰기와 BPE로 인한 띄어쓰기를 구분해야 할 필요성이 있습니다. 그래서 특수문자(기존의 _가 아닌 ▁)를 사용하여 기존의 띄어쓰기(또는 단어의 시작)를 나타냅니다. 이를 통해 후에 설명할 분절 복원에서 문장을 BPE이전의 형태로 원상복구 할 수 있습니다.

한글 Mecab에 의해 분절 된 원문
```
자연어 처리 는 인공지능 의 한 줄기 입니다 .
시퀀스 투 시퀀스 의 등장 이후 로 딥 러닝 을 활용 한 자연어 처리 는 새로운 전기 를 맞이 하 게 되 었 습니다 .
문장 을 받 아 단순히 수치 로 나타내 던 시절 을 넘 어 , 원 하 는 대로 문장 을 만들 어 낼 수 있 게 된 것 입니다 .
이 에 따라 이전 까지 큰 변화 가 없 었 던 자연어 처리 분야 의 연구 는 폭발 적 으로 늘어나 기 시작 하 여 , 곧 기계 번역 시스템 은 신경망 에 의해 정복 당하 였 습니다 .
또한 , attention 기법 의 고도 화 로 전이 학습 이 발전 하 면서 , QA 문제 도 사람 보다 정확 한 수준 이 되 었 습니다 .
```

한글 BPE 적용
```
▁자연 어 ▁처리 ▁는 ▁인공지능 ▁의 ▁한 ▁줄기 ▁입니다 ▁.
▁시퀀스 ▁투 ▁시퀀스 ▁의 ▁등장 ▁이후 ▁로 ▁딥 ▁러 닝 ▁을 ▁활용 ▁한 ▁자연 어 ▁처리 ▁는 ▁새로운 ▁전기 ▁를 ▁맞이 ▁하 ▁게 ▁되 ▁었 ▁습니다 ▁.
▁문장 ▁을 ▁받 ▁아 ▁단순히 ▁수치 ▁로 ▁나타내 ▁던 ▁시절 ▁을 ▁넘 ▁어 ▁, ▁원 ▁하 ▁는 ▁대로 ▁문장 ▁을 ▁만들 ▁어 ▁낼 ▁수 ▁있 ▁게 ▁된 ▁것 ▁입니다 ▁.
▁이 ▁에 ▁따라 ▁이전 ▁까지 ▁큰 ▁변화 ▁가 ▁없 ▁었 ▁던 ▁자연 어 ▁처리 ▁분야 ▁의 ▁연구 ▁는 ▁폭발 ▁적 ▁으로 ▁늘어나 ▁기 ▁시작 ▁하 ▁여 ▁, ▁곧 ▁기계 ▁번역 ▁시스템 ▁은 ▁신경 망 ▁에 ▁의해 ▁정복 ▁당하 ▁였 ▁습니다 ▁.
▁또한 ▁, ▁attention ▁기법 ▁의 ▁고도 ▁화 ▁로 ▁전이 ▁학습 ▁이 ▁발전 ▁하 ▁면서 ▁, ▁Q A ▁문제 ▁도 ▁사람 ▁보다 ▁정확 ▁한 ▁수준 ▁이 ▁되 ▁었 ▁습니다 ▁.
```

영어 NLTK에 의해 분절 된 원문
```
Natural language processing is one of biggest streams in artificial intelligence , and it becomes very popular after seq2seq 's invention .
However , in order to make a strong A.I . , there are still many challenges remain .
I believe that we can breakthrough these barriers to get strong artificial intelligence .
```

영어 BPE 적용
```
▁Natural ▁language ▁processing ▁is ▁one ▁of ▁biggest ▁stream s ▁in ▁artificial ▁intelligence ▁, ▁and ▁it ▁becomes ▁very ▁popular ▁after ▁se q 2 se q ▁'s ▁invention ▁.
▁However ▁, ▁in ▁order ▁to ▁make ▁a ▁strong ▁A. I ▁. ▁, ▁there ▁are ▁still ▁many ▁challenges ▁remain ▁.
▁I ▁believe ▁that ▁we ▁can ▁breakthrough ▁these ▁barriers ▁to ▁get ▁strong ▁artificial ▁intelligence ▁.
```

|원문|서브워드 분절 결과|
|-|-|
|러닝|러 + 닝|
|신경망|신경 + 망|
|자연어|자연 + 어|
|seq2seq|se + q + 2 + se + q|
|streams|stream + s|

위와 같이 서브워드 분절에 의해서 추가적으로 쪼개진 단어들은 적절한 서브워드의 형태로 나누어진 것을 볼 수 있습니다. 특히, 숫자 같은 경우에는 자주 쓰이는 숫자 50이나 ,000의 경우 성공적으로 하나의 서브워드로 지정된 것을 볼 수 있습니다.

## 오픈 소스

서브워드 단위 분절을 수행하기 위한 오픈소스들은 다음과 같습니다. 구글의 SentencePiece 모듈이 속도가 빠르지만, 논문 원저자의 파이썬 코드는 수정이 쉬워 편의에 따라 사용하면 됩니다.

- Sennrich(원저자)의 깃허브: https://github.com/rsennrich/subword-nmt
- 본서의 저자가 수정한 버전: https://github.com/kh-kim/subword-nmt
- Google의 SentencePiece 모듈: https://github.com/google/sentencepiece
