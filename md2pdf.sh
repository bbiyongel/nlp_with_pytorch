FONT_SIZE=12pt
DIR_PATH=./pdf/

# * [소개글](README.md)
# * [Index](index_list.md)
echo 'cover'
cd ./nlp-with-deeplearning
pandoc ../README.md ../index_list.md --latex-engine=xelatex -o ../${DIR_PATH}/0.cover.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [딥러닝을 활용한 자연어처리](nlp-with-deeplearning/cover.md)
#   * [서문](nlp-with-deeplearning/intro.md)
#   * [딥러닝의 역사](nlp-with-deeplearning/deeplearning.md)
#   * [왜 자연어처리는 어려울까](nlp-with-deeplearning/why-nlp-difficult.md)
#   * [왜 한국어 자연어처리는 더욱 어려울까](nlp-with-deeplearning/korean-is-hell.md)
#   * [최근 추세](nlp-with-deeplearning/trends.md)
echo 'nlp-with-deeplearning'
cd ./nlp-with-deeplearning
pandoc ./cover.md ./intro.md ./deeplearning.md ./why-nlp-difficult.md ./korean-is-hell.md ./trends.md --latex-engine=xelatex -o ../${DIR_PATH}/1.introduction.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [기초 수학](basic-math/cover.md)
#   * [서문](basic-math/intro.md)
#   * [랜덤 변수와 확률 분포](basic-math/prob-dist.md)
#   * [쉬어가기: 몬티홀 문제](basic-math/monty-hall.md)
#   * [기대값과 샘플링](basic-math/sampling.md)
#   * [Maximum Likelihood Estimation](basic-math/mle.md)
#   * [정보이론](basic-math/information.md)
#   * [쉬어가기: Mean Square Error (MSE)](basic-math/mse.md)
#   * [결론](basic-math/conclusion.md)
echo 'basic-math'
cd ./basic-math
pandoc ./cover.md ./intro.md ./prob-dist.md ./monty-hall.md ./sampling.md ./mle.md ./information.md ./mse.md ./conclusion.md  --latex-engine=xelatex -o ../${DIR_PATH}/2.basic-math.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [Hello PyTorch](pytorch-intro/cover.md)
#   * [소개](pytorch-intro/intro.md)
#   * [설치 방법](pytorch-intro/how-to-install.md)
#   * [PyTorch 짧은 튜토리얼](pytorch-intro/hello-pytorch.md)
echo 'pytorch-intro'
cd ./pytorch-intro
pandoc ./cover.md ./intro.md ./how-to-install.md ./hello-pytorch.md --latex-engine=xelatex -o ../${DIR_PATH}/3.pytorch-intro.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [전처리](preprocessing/cover.md)
#   * [서문](preprocessing/intro.md)
#   * [코퍼스 수집](preprocessing/collecting-corpus.md)
#   * [코퍼스 정제](preprocessing/cleaning-corpus.md)
#   * [분절하기 (형태소 분석)](preprocessing/tokenization.md)
#   * [병렬 코퍼스 만들기](preprocessing/align.md)
#   * [서브워드 분절하기](preprocessing/bpe.md)
#   * [분절 복구하기](preprocessing/detokenization.md)
#   * [TorchText](preprocessing/torchtext.md)
echo 'preprocessing'
cd ./preprocessing
pandoc ./cover.md ./intro.md ./collecting-corpus.md ./cleaning-corpus.md ./tokenization.md ./align.md ./bpe.md ./detokenization.md ./torchtext.md --latex-engine=xelatex -o ../${DIR_PATH}/4.preprocessing.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [의미: 유사성과 모호성](word-senses/cover.md)
#   * [소개](word-senses/intro.md)
#   * [One-hot 인코딩](word-senses/one-hot-encoding.md)
#   * [워드넷](word-senses/wordnet.md)
#   * [출현빈도 활용하기: TF-IDF](word-senses/tf-idf.md)
#   * [유사도 구하기](word-senses/similarity.md)
#   * [단어 중의성 해소](word-senses/wsd.md)
#   * [Selectional Preference](word-senses/selectional-preference.md)
#   * [결론](word-senses/conclusion.md)
echo 'word-senses'
cd ./word-senses
pandoc ./cover.md ./one-hot-encoding.md ./wordnet.md ./tf-idf.md ./similarity.md ./wsd.md ./selectional-preference.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/5.word-senses.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [워드 임베딩](word-embedding-vector/cover.md)
#   * [서문](word-embedding-vector/intro.md)
#   * [차원 축소](word-embedding-vector/dimension-reduction.md)
#   * [흔한 오해](word-embedding-vector/myth.md)
#   * [Word2Vec](word-embedding-vector/word2vec.md)
#   * [GloVe](word-embedding-vector/glove.md)
#   * [예제 코드](word-embedding-vector/example.md)
echo 'word-embedding-vector'
cd ./word-embedding-vector
pandoc ./cover.md ./intro.md ./dimension-reduction.md ./myth.md ./word2vec.md ./glove.md ./example.md --latex-engine=xelatex -o ../${DIR_PATH}/6.word-embedding.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [시퀀스 모델링](sequential-modeling/cover.md)
#   * [서문](sequential-modeling/intro.md)
#   * [Recurrent Neural Network](sequential-modeling/rnn.md)
#   * [Long Short Term Memory](sequential-modeling/lstm.md)
#   * [Gated Recurrent Unit](sequential-modeling/gru.md)
#   * [그래디언트 clipping](sequential-modeling/gradient-clipping.md)
echo 'sequential-modeling'
cd ./sequential-modeling
pandoc ./cover.md ./intro.md ./rnn.md ./lstm.md ./gru.md ./gradient-clipping.md --latex-engine=xelatex -o ../${DIR_PATH}/7.sequential-modeling.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [텍스트 분류](text-classification/cover.md)
#   * [서문](text-classification/intro.md)
#   * [Naive Bayes](text-classification/naive-bayes.md)
#   * [RNN을 활용한 텍스트 분류](text-classification/rnn.md)
#   * [CNN을 활용한 텍스트 분류](text-classification/cnn.md)
#   * [구현 예제](text-classification/code.md)
echo 'text-classification'
cd ./text-classification
pandoc ./cover.md ./intro.md ./naive-bayes.md ./cnn.md ./rnn.md ./code.md --latex-engine=xelatex -o ../${DIR_PATH}/8.text-classification.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [언어 모델링](language-modeling/cover.md)
#   * [서문](language-modeling/intro.md)
#   * [n-gram](language-modeling/n-gram.md)
#   * [Perpexity](language-modeling/perpexity.md)
#   * [n-gram 예제](language-modeling/srilm.md)
#   * [뉴럴네트워크 언어 모델링](language-modeling/nnlm.md)
#   * [활용 분야](language-modeling/application.md)
echo 'language-modeling'
cd ./language-modeling
pandoc ./cover.md ./intro.md ./n-gram.md ./perpexity.md ./srilm.md ./nnlm.md ./application.md --latex-engine=xelatex -o ../${DIR_PATH}/9.language-modeling.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [신경망 기계번역](neural-machine-translation/cover.md)
#   * [서문](neural-machine-translation/intro.md)
#   * [Sequence-to-Sequence](neural-machine-translation/seq2seq.md)
#   * [Attention](neural-machine-translation/attention.md)
#   * [Input Feeding](neural-machine-translation/input-feeding.md)
#   * [Auto-regressive and Teacher Forcing](neural-machine-translation/teacher-forcing.md)
#   * [탐색(추론)](neural-machine-translation/beam-search.md)
#   * [성능 평가 방법](neural-machine-translation/eval.md)
#   * [구현 예제](neural-machine-translation/code.md)
echo 'machine-translation'
cd ./neural-machine-translation
pandoc ./cover.md ./intro.md ./seq2seq.md ./attention.md ./input-feeding.md ./teacher-forcing.md beam-search.md eval.md code.md --latex-engine=xelatex -o ../${DIR_PATH}/10.nmt.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [신경망 기계 번역 심화 주제](adv-nmt/cover.md)
#   * [다국어 번역](adv-nmt/multilingual-nmt.md)
#   * [단방향 코퍼스를 활용하기](adv-nmt/monolingual-corpus.md)
#   * [트랜스포머](adv-nmt/transformer.md)
echo 'adv-nmt'
cd ./adv-nmt
pandoc ./cover.md ./multilingual-nmt.md ./monolingual-corpus.md transformer.md --latex-engine=xelatex -o ../${DIR_PATH}/11.adv-nmt.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=12pt
cd ..

# * [강화학습을 활용한 자연어생성](reinforcement-learning/cover.md)
#   * [서문](reinforcement-learning/intro.md)
#   * [강화학습 기초](reinforcement-learning/rl_basics.md)
#   * [폴리시 그래디언트](reinforcement-learning/policy-gradient.md)
#   * [자연어생성에서의 강화학습의 특성](reinforcement-learning/characteristic.md)
#   * [강화학습을 활용한 지도학습](reinforcement-learning/supervised-nmt.md)
#   * [강화학습을 활용한 비지도학습](reinforcement-learning/unsupervised-nmt.md)
echo 'rl'
cd ./reinforcement-learning
pandoc ./cover.md ./intro.md ./rl_basics.md ./policy-gradient.md ./characteristic.md ./supervised-nmt.md ./unsupervised-nmt.md --latex-engine=xelatex -o ../${DIR_PATH}/12.rl.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [듀얼리티 활용하기](duality/cover.md)
#   * [듀얼리티란](duality/intro.md)
#   * [듀얼리티를 활용한 지도학습](duality/dsl.md)
#   * [듀얼리티를 활용한 비지도학습](duality/dul.md)
echo 'duality'
cd ./duality
pandoc ./cover.md ./intro.md ./dsl.md ./dul.md --latex-engine=xelatex -o ../${DIR_PATH}/13.duality.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..

# * [서비스 만들기](productization/cover.md)
#   * [파이프라인](productization/pipeline.md)
#   * [Google의 신경망 기계번역](productization/gnmt.md)
#   * [Edinburgh 대학의 신경망 기계번역](productization/nematus.md)
#   * [Microsoft의 신경망 기계번역](productization/microsoft.md)
echo 'productization'
cd ./productization
pandoc ./cover.md ./pipeline.md ./gnmt.md ./nematus.md ./microsoft.md --latex-engine=xelatex -o ../${DIR_PATH}/14.productization.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..