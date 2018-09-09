# Using Other Features for Similarity

이번엔 다른 방식으로 단어 중의성 해소(WSD)에 접근 해보겠습니다. 자체적으로 단어에 대한 특성(feature)들을 모아 feature vector로 만들거나 유사도(similarity)를 계산하는 연산을 통해 단어의 중의성을 해소하는 방법입니다. 지금이야 어렵지않게 단어를 vector 형태로 embedding 할 수 있지만, 딥러닝 이전의 시대에는 쉽지 않은 일이었습니다. 다른 word embedding 방식은 추후 Word Embedding Vector 챕터에서 다루도록 하고, 중의성 해소를 위한 피쳐를 추출하고 유사도를 계산하는 방식을 살펴보겠습니다.

## Based on Co-Occurrence

http://www.let.rug.nl/nerbonne/teach/rema-stats-meth-seminar/presentations/Olango-Naive-Bayes-2009.pdf