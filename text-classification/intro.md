# Text Classification

Text Classification(텍스트 분류)은 텍스트, 문장 또는 문서(문장들)를 입력으로 받아 사전에 정의된 클래스(class)들 중에서 어떤 클래스에 속하는지 분류 하는 과정을 의미합니다. 따라서 Text Classification은 어쩌면 이 책에서 (그 난이도에 비해서) 독자들에게 가장 쓸모가 있는 챕터가 될 수도 있습니다. Text Classificaion의 응용 분야가 다양하기 때문 입니다.

|문제|클래스 예|
|---|---|
|감성분석(Sentiment Analysis)|긍정(positive), 중립(neutral), 부정(negative)|
|스팸 메일 탐지(Spam E-mail Detection)|정상(normal), 스팸(spam)|
|사용자 의도 분류(User Intent Classificaion)|명령, 질문, 잡담 등
|주제 분류|각 주제|
|카테고리 분류|각 카테고리|

등 무언가 분류해야 하는 문제가 있다면 대부분 text classificaion에 속한다고 볼 수 있습니다. 딥러닝 이전에는 Naive Bayes, SVM 등 다양한 방법이 존재하였습니다. 이번 챕터에서는 딥러닝 이전의 가장 간단한 방식인 naive bayes 방식과 딥러닝 방식들을 소개 하도록 하겠습니다.