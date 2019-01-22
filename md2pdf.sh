FONT_SIZE=11pt
DIR_PATH=./pdf/
MARGIN=1in
LINE_SPACE=1.7

# * [소개글](README.md)
# * [Index](index_list.md)
echo 'cover'
cd ./nlp-with-deeplearning
time pandoc ../README.md ../index_list.md --latex-engine=xelatex -o ../${DIR_PATH}/0.cover.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

echo 'preface'
cd ./nlp-with-deeplearning
time pandoc ../preface.md --latex-engine=xelatex -o ../${DIR_PATH}/0.preface.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [딥러닝을 활용한 자연어처리](1-introduction/cover.md)
#   * [서문](1-introduction/intro.md)
#   * [딥러닝의 역사](1-introduction/deeplearning.md)
#   * [왜 자연어처리는 어려울까](1-introduction/why-nlp-difficult.md)
#   * [왜 한국어 자연어처리는 더욱 어려울까](1-introduction/korean-is-hell.md)
#   * [최근 추세](1-introduction/trends.md)
echo 'nlp-with-deeplearning'
cd ./nlp-with-deeplearning
time pandoc ./cover.md ./intro.md ./deeplearning.md ./why-nlp-difficult.md ./korean-is-hell.md ./trends.md --latex-engine=xelatex -o ../${DIR_PATH}/1.introduction.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [기초 수학](2-basic_math/cover.md)
#   * [서문](2-basic_math/intro.md)
#   * [랜덤 변수와 확률 분포](2-basic_math/prob-dist.md)
#   * [쉬어가기: 몬티홀 문제](2-basic_math/monty-hall.md)
#   * [기대값과 샘플링](2-basic_math/sampling.md)
#   * [Maximum Likelihood Estimation](2-basic_math/mle.md)
#   * [정보이론](2-basic_math/information.md)
#   * [쉬어가기: Mean Square Error (MSE)](2-basic_math/mse.md)
#   * [정리](2-basic_math/conclusion.md)
echo 'basic-math'
cd ./basic-math
time pandoc ./cover.md ./intro.md ./prob-dist.md ./monty-hall.md ./sampling.md ./mle.md ./information.md ./mse.md ./conclusion.md  --latex-engine=xelatex -o ../${DIR_PATH}/2.basic-math.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [Hello PyTorch](3-pytorch_tutorial/cover.md)
#   * [소개](3-pytorch_tutorial/intro.md)
#   * [설치 방법](3-pytorch_tutorial/how-to-install.md)
#   * [PyTorch 짧은 튜토리얼](3-pytorch_tutorial/hello-pytorch.md)
echo 'pytorch-intro'
cd ./pytorch-intro
time pandoc ./cover.md ./intro.md ./how-to-install.md ./hello-pytorch.md --latex-engine=xelatex -o ../${DIR_PATH}/3.pytorch-intro.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [전처리](4-preprocessing/cover.md)
#   * [들어가기에 앞서](4-preprocessing/intro.md)
#   * [코퍼스 수집](4-preprocessing/collecting-corpus.md)
#   * [코퍼스 정제](4-preprocessing/cleaning-corpus.md)
#   * [분절하기 (형태소 분석)](4-preprocessing/tokenization.md)
#   * [병렬 코퍼스 만들기](4-preprocessing/align.md)
#   * [서브워드 분절하기](4-preprocessing/bpe.md)
#   * [분절 복원하기](4-preprocessing/detokenization.md)
#   * [토치텍스트](4-preprocessing/torchtext.md)
echo 'preprocessing'
cd ./preprocessing
time pandoc ./cover.md ./intro.md ./collecting-corpus.md ./cleaning-corpus.md ./tokenization.md ./align.md ./bpe.md ./detokenization.md ./torchtext.md --latex-engine=xelatex -o ../${DIR_PATH}/4.preprocessing.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [의미: 유사성과 모호성](5-word_senses/cover.md)
#   * [소개](5-word_senses/intro.md)
#   * [One-hot 인코딩](5-word_senses/one-hot-encoding.md)
#   * [워드넷](5-word_senses/wordnet.md)
#   * [피쳐란](5-word_senses/feature.md)
#   * [피쳐 추출하기: TF-IDF](5-word_senses/tf-idf.md)
#   * [피쳐 벡터 만들기](5-word_senses/vectorization.md)
#   * [벡터 유사도 구하기](5-word_senses/similarity.md)
#   * [단어 중의성 해소](5-word_senses/wsd.md)
#   * [Selectional Preference](5-word_senses/selectional-preference.md)
#   * [정리](5-word_senses/conclusion.md)
echo 'word-senses'
cd ./word-senses
time pandoc ./cover.md ./intro.md ./one-hot-encoding.md ./wordnet.md ./feature.md ./tf-idf.md ./vectorization.md ./similarity.md ./wsd.md ./selectional-preference.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/5.word-senses.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [워드 임베딩](6-word_embedding/cover.md)
#   * [서문](6-word_embedding/intro.md)
#   * [차원 축소](6-word_embedding/dimension-reduction.md)
#   * [흔한 오해](6-word_embedding/myth.md)
#   * [Word2Vec](6-word_embedding/word2vec.md)
#   * [GloVe](6-word_embedding/glove.md)
#   * [예제 코드](6-word_embedding/example.md)
#   * [정리](6-word_embedding/conclusion.md)
echo 'word-embedding-vector'
cd ./word-embedding-vector
time pandoc ./cover.md ./intro.md ./dimension-reduction.md ./myth.md ./word2vec.md ./glove.md ./example.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/6.word-embedding.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [시퀀스 모델링](7-sequential_modeling/cover.md)
#   * [서문](7-sequential_modeling/intro.md)
#   * [Recurrent Neural Network](7-sequential_modeling/rnn.md)
#   * [Long Short Term Memory](7-sequential_modeling/lstm.md)
#   * [Gated Recurrent Unit](7-sequential_modeling/gru.md)
#   * [그래디언트 클리핑](7-sequential_modeling/gradient-clipping.md)
#   * [정리](7-sequential_modeling/conclusion.md)
echo 'sequential-modeling'
cd ./sequential-modeling
time pandoc ./cover.md ./intro.md ./rnn.md ./lstm.md ./gru.md ./gradient-clipping.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/7.sequential-modeling.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [텍스트 분류](8-text_classification/cover.md)
#   * [서문](8-text_classification/intro.md)
#   * [나이브 베이즈를 활용하기](8-text_classification/naive-bayes.md)
#   * [흔한 오해 2](8-text_classification/myth.md)
#   * [RNN을 활용하기](8-text_classification/rnn.md)
#   * [CNN을 활용하기](8-text_classification/cnn.md)
#   * [정리](8-text_classification/conclusion.md)
echo 'text-classification'
cd ./text-classification
time pandoc ./cover.md ./intro.md ./naive-bayes.md ./myth.md ./rnn.md ./cnn.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/8.text-classification.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [언어 모델링](9-language_modeling/cover.md)
#   * [서문](9-language_modeling/intro.md)
#   * [n-gram](9-language_modeling/n-gram.md)
#   * [Perpexity](9-language_modeling/perpexity.md)
#   * [n-gram 예제](9-language_modeling/srilm.md)
#   * [뉴럴네트워크 언어 모델링](9-language_modeling/nnlm.md)
#   * [활용 분야](9-language_modeling/application.md)
#   * [정리](9-language_modeling/conclusion.md)
echo 'language-modeling'
cd ./language-modeling
time pandoc ./cover.md ./intro.md ./n-gram.md ./perpexity.md ./srilm.md ./nnlm.md ./application.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/9.language-modeling.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [신경망 기계번역](10-neural_machine_translation/cover.md)
#   * [서문](10-neural_machine_translation/intro.md)
#   * [Sequence-to-Sequence](10-neural_machine_translation/seq2seq.md)
#   * [Attention](10-neural_machine_translation/attention.md)
#   * [Input Feeding](10-neural_machine_translation/input-feeding.md)
#   * [Auto-regressive 속성과 Teacher Forcing](10-neural_machine_translation/teacher-forcing.md)
#   * [탐색(추론)](10-neural_machine_translation/beam-search.md)
#   * [성능 평가 방법](10-neural_machine_translation/eval.md)
#   * [정리](10-neural_machine_translation/conclusion.md)
echo 'machine-translation'
cd ./neural-machine-translation
time pandoc ./cover.md ./intro.md ./seq2seq.md ./attention.md ./input-feeding.md ./teacher-forcing.md ./beam-search.md ./eval.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/10.nmt.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [신경망 기계번역 심화 주제](adv-nmt/cover.md)
#   * [다국어 번역](adv-nmt/multilingual-nmt.md)
#   * [단방향 코퍼스를 활용하기](adv-nmt/monolingual-corpus.md)
#   * [트랜스포머](adv-nmt/transformer.md)
#   * [정리](adv-nmt/conclusion.md)
echo 'adv-nmt'
cd ./adv-nmt
time pandoc ./cover.md ./multilingual-nmt.md ./monolingual-corpus.md transformer.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/11.adv-nmt.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [강화학습을 활용한 자연어생성](reinforcement-learning/cover.md)
#   * [서문](reinforcement-learning/intro.md)
#   * [강화학습 기초](reinforcement-learning/rl_basics.md)
#   * [폴리시 그래디언트](reinforcement-learning/policy-gradient.md)
#   * [자연어생성에서의 강화학습의 특성](reinforcement-learning/characteristic.md)
#   * [강화학습을 활용한 지도학습](reinforcement-learning/supervised-nmt.md)
#   * [강화학습을 활용한 비지도학습](reinforcement-learning/unsupervised-nmt.md)
#   * [정리](reinforcement-learning/conclusion.md)
echo 'rl'
cd ./reinforcement-learning
time pandoc ./cover.md ./intro.md ./rl_basics.md ./policy-gradient.md ./characteristic.md ./supervised-nmt.md ./unsupervised-nmt.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/12.rl.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [듀얼리티 활용하기](duality/cover.md)
#   * [듀얼리티란](duality/intro.md)
#   * [듀얼리티를 활용한 지도학습](duality/dsl.md)
#   * [듀얼리티를 활용한 비지도학습](duality/dul.md)
#   * [쉬어가기: Back-translation을 재해석 하기](duality/back_translation.md)
#   * [정리](duality/conclusion.md)
echo 'duality'
cd ./duality
time pandoc ./cover.md ./intro.md ./dsl.md ./dul.md ./back_translation.md ./conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/13.duality.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [서비스 만들기](14_productization/cover.md)
#   * [파이프라인](14_productization/pipeline.md)
#   * [Google의 신경망 기계번역](14_productization/gnmt.md)
#   * [Edinburgh 대학의 신경망 기계번역](14_productization/nematus.md)
#   * [Microsoft의 신경망 기계번역](14_productization/microsoft.md)
echo 'productization'
cd ./productization
time pandoc ./cover.md ./pipeline.md ./gnmt.md ./nematus.md ./microsoft.md --latex-engine=xelatex -o ../${DIR_PATH}/14.productization.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

echo 'epilogue'
cd ./nlp-with-deeplearning
time pandoc ../epilogue.md --latex-engine=xelatex -o ../${DIR_PATH}/0.epilogue.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

time gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=./pdf/merged.pdf ./pdf/0.cover.pdf ./pdf/0.preface.pdf ./pdf/1.introduction.pdf ./pdf/2.basic-math.pdf ./pdf/3.pytorch-intro.pdf ./pdf/4.preprocessing.pdf ./pdf/5.word-senses.pdf ./pdf/6.word-embedding.pdf ./pdf/7.sequential-modeling.pdf ./pdf/8.text-classification.pdf ./pdf/9.language-modeling.pdf ./pdf/10.nmt.pdf ./pdf/11.adv-nmt.pdf ./pdf/12.rl.pdf ./pdf/13.duality.pdf ./pdf/14.productization.pdf ./pdf/0.epilogue.pdf

python ./collect_info.py ./ image_needed.jpeg > ./image_needed.txt
