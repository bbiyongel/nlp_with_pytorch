# Text Classification

Text Classification은 텍스트 문장 또는 문서(문장들)를 입력으로 받아 사전에 정의된 class 들 중에서 어떤 class에 속하는지 분류 하는 과정을 의미합니다. 따라서 Text Classification은 어쩌면 이 책에서 (그 난이도에 비해서) 독자들에게 가장 쓸모가 있는 챕터가 될 지도 모릅니다. Text Classificaion의 응용분야가 다양하기 때문입니다.

- Sentiment Analysis (감성분석)
- Spam E-mail Detection (스팸 메일 감지)
- User Intent Classificaion (사용자 의도 파악)
- Topic Classificaion (주제 분류)
- Categorization (카테고리 분류)

등 무언가 분류해야 하는 task가 있다면 대부분 text classificaion에 속한다고 볼 수 있습니다. 딥러닝 이전에는 Navie Bayes, LDA, SVM, SVD, PCA 등 다양한 방법이 존재하였습니다. 이번 챕터에서는 딥러닝 이전의 가장 간단한 방식인 naive bayes 방식과 딥러닝 방식들을 소개 하도록 하겠습니다.