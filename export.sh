FONT_SIZE=11pt
DIR_PATH=./export/
MARGIN=1in
LINE_SPACE=1.7

# * [소개글](README.md)
# * [서문](preface.md)
# * [Index](index_list.md)
echo 'preface'
cd ./01-introduction
time pandoc ../preface.md ../index_list.md --latex-engine=xelatex -o ../${DIR_PATH}/0.preface.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ../preface.md ../index_list.md --latex-engine=xelatex -o ../${DIR_PATH}/0.preface.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [딥러닝을 활용한 자연어처리](01-introduction/00-cover.md)
#   * [서문](01-introduction/01-intro.md)
#   * [딥러닝의 역사](01-introduction/02-deeplearning.md)
#   * [왜 자연어처리는 어려울까](01-introduction/03-why-nlp-difficult.md)
#   * [왜 한국어 자연어처리는 더욱 어려울까](01-introduction/04-korean-is-hell.md)
#   * [최근 추세](01-introduction/05-trends.md)
echo '01-introduction'
cd ./01-introduction
time pandoc ./00-cover.md ./01-intro.md ./02-deeplearning.md ./03-why-nlp-difficult.md ./04-korean-is-hell.md ./05-trends.md --latex-engine=xelatex -o ../${DIR_PATH}/1.introduction.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-deeplearning.md ./03-why-nlp-difficult.md ./04-korean-is-hell.md ./05-trends.md --latex-engine=xelatex -o ../${DIR_PATH}/1.introduction.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [기초 수학](02-basic_math/00-cover.md)
#   * [서문](02-basic_math/01-intro.md)
#   * [랜덤 변수와 확률 분포](02-basic_math/02-prob-dist.md)
#   * [쉬어가기: 몬티홀 문제](02-basic_math/03-monty-hall.md)
#   * [기대값과 샘플링](02-basic_math/04-sampling.md)
#   * [Maximum Likelihood Estimation](02-basic_math/05-mle.md)
#   * [정보이론](02-basic_math/06-information.md)
#   * [쉬어가기: Mean Square Error (MSE)](02-basic_math/07-mse.md)
#   * [정리](02-basic_math/08-conclusion.md)
echo '02-basic_math'
cd ./02-basic_math
time pandoc ./00-cover.md ./01-intro.md ./02-prob-dist.md ./03-monty-hall.md ./04-sampling.md ./05-mle.md ./06-information.md ./07-mse.md ./08-conclusion.md  --latex-engine=xelatex -o ../${DIR_PATH}/2.basic-math.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-prob-dist.md ./03-monty-hall.md ./04-sampling.md ./05-mle.md ./06-information.md ./07-mse.md ./08-conclusion.md  --latex-engine=xelatex -o ../${DIR_PATH}/2.basic-math.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [Hello PyTorch](03-pytorch_tutorial/00-cover.md)
#   * [준비](03-pytorch_tutorial/01-intro.md)
#   * [소개 및 설치](03-pytorch_tutorial/02-how-to-install.md)
#   * [짧은 튜토리얼](03-pytorch_tutorial/03-hello-pytorch.md)
echo '03-pytorch_tutorial'
cd ./03-pytorch_tutorial
time pandoc ./00-cover.md ./01-intro.md ./02-how-to-install.md ./03-hello-pytorch.md --latex-engine=xelatex -o ../${DIR_PATH}/3.pytorch-intro.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-how-to-install.md ./03-hello-pytorch.md --latex-engine=xelatex -o ../${DIR_PATH}/3.pytorch-intro.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [전처리](04-preprocessing/00-cover.md)
#   * [들어가기에 앞서](04-preprocessing/01-intro.md)
#   * [코퍼스 수집](04-preprocessing/02-collecting-corpus.md)
#   * [코퍼스 정제](04-preprocessing/03-cleaning-corpus.md)
#   * [분절하기 (형태소 분석)](04-preprocessing/04-tokenization.md)
#   * [병렬 코퍼스 만들기](04-preprocessing/05-align.md)
#   * [서브워드 분절하기](04-preprocessing/06-bpe.md)
#   * [분절 복원하기](04-preprocessing/07-detokenization.md)
#   * [토치텍스트](04-preprocessing/08-torchtext.md)
echo '04-preprocessing'
cd ./04-preprocessing
time pandoc ./00-cover.md ./01-intro.md ./02-collecting-corpus.md ./03-cleaning-corpus.md ./04-tokenization.md ./05-align.md ./06-bpe.md ./07-detokenization.md ./08-torchtext.md --latex-engine=xelatex -o ../${DIR_PATH}/4.preprocessing.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-collecting-corpus.md ./03-cleaning-corpus.md ./04-tokenization.md ./05-align.md ./06-bpe.md ./07-detokenization.md ./08-torchtext.md --latex-engine=xelatex -o ../${DIR_PATH}/4.preprocessing.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [의미: 유사성과 모호성](05-word_senses/00-cover.md)
#   * [소개](05-word_senses/01-intro.md)
#   * [One-hot 인코딩](05-word_senses/02-one-hot-encoding.md)
#   * [워드넷](05-word_senses/03-wordnet.md)
#   * [피쳐란](05-word_senses/04-feature.md)
#   * [피쳐 추출하기: TF-IDF](05-word_senses/05-tf-idf.md)
#   * [피쳐 벡터 만들기](05-word_senses/06-vectorization.md)
#   * [벡터 유사도 구하기](05-word_senses/07-similarity.md)
#   * [단어 중의성 해소](05-word_senses/08-wsd.md)
#   * [Selectional Preference](05-word_senses/09-selectional-preference.md)
#   * [정리](05-word_senses/10-conclusion.md)
echo '05-word_senses'
cd ./05-word_senses
time pandoc ./00-cover.md ./01-intro.md ./02-one-hot-encoding.md ./03-wordnet.md ./04-feature.md ./05-tf-idf.md ./06-vectorization.md ./07-similarity.md ./08-wsd.md ./09-selectional-preference.md ./10-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/5.word-senses.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-one-hot-encoding.md ./03-wordnet.md ./04-feature.md ./05-tf-idf.md ./06-vectorization.md ./07-similarity.md ./08-wsd.md ./09-selectional-preference.md ./10-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/5.word-senses.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [워드 임베딩](06-word_embedding/00-cover.md)
#   * [서문](06-word_embedding/01-intro.md)
#   * [차원 축소](06-word_embedding/02-dimension-reduction.md)
#   * [흔한 오해](06-word_embedding/03-myth.md)
#   * [Word2Vec](06-word_embedding/04-word2vec.md)
#   * [GloVe](06-word_embedding/05-glove.md)
#   * [예제](06-word_embedding/06-example.md)
#   * [정리](06-word_embedding/07-conclusion.md)
echo '06-word_embedding'
cd ./06-word_embedding
time pandoc ./00-cover.md ./01-intro.md ./02-dimension-reduction.md ./03-myth.md ./04-word2vec.md ./05-glove.md ./06-example.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/6.word-embedding.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-dimension-reduction.md ./03-myth.md ./04-word2vec.md ./05-glove.md ./06-example.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/6.word-embedding.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [시퀀스 모델링](07-sequential_modeling/00-cover.md)
#   * [서문](07-sequential_modeling/01-intro.md)
#   * [Recurrent Neural Network](07-sequential_modeling/02-rnn.md)
#   * [Long Short Term Memory](07-sequential_modeling/03-lstm.md)
#   * [Gated Recurrent Unit](07-sequential_modeling/04-gru.md)
#   * [그래디언트 클리핑](07-sequential_modeling/05-gradient-clipping.md)
#   * [정리](07-sequential_modeling/06-conclusion.md)
echo '07-sequential_modeling'
cd ./07-sequential_modeling
time pandoc ./00-cover.md ./01-intro.md ./02-rnn.md ./03-lstm.md ./04-gru.md ./05-gradient-clipping.md ./06-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/7.sequential-modeling.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-rnn.md ./03-lstm.md ./04-gru.md ./05-gradient-clipping.md ./06-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/7.sequential-modeling.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [텍스트 분류](08-text_classification/00-cover.md)
#   * [서문](08-text_classification/01-intro.md)
#   * [나이브 베이즈를 활용하기](08-text_classification/02-naive-bayes.md)
#   * [흔한 오해 2](08-text_classification/03-myth.md)
#   * [RNN을 활용하기](08-text_classification/04-rnn.md)
#   * [CNN을 활용하기](08-text_classification/05-cnn.md)
#   * [쉬어가기: 멀티 레이블 분류](08-text_classification/06-multi_classification.md)
#   * [정리](08-text_classification/07-conclusion.md)
echo '08-text_classification'
cd ./08-text_classification
time pandoc ./00-cover.md ./01-intro.md ./02-naive-bayes.md ./03-myth.md ./04-rnn.md ./05-cnn.md ./06-multi_classification.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/8.text-classification.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-naive-bayes.md ./03-myth.md ./04-rnn.md ./05-cnn.md ./06-multi_classification.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/8.text-classification.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [언어 모델링](09-language_modeling/00-cover.md)
#   * [서문](09-language_modeling/01-intro.md)
#   * [n-gram](09-language_modeling/02-n-gram.md)
#   * [Perpexity](09-language_modeling/03-perpexity.md)
#   * [n-gram 예제](09-language_modeling/04-srilm.md)
#   * [뉴럴네트워크 언어 모델링](09-language_modeling/05-nnlm.md)
#   * [활용 분야](09-language_modeling/06-application.md)
#   * [정리](09-language_modeling/07-conclusion.md)
echo '09-language_modeling'
cd ./09-language_modeling
time pandoc ./00-cover.md ./01-intro.md ./02-n-gram.md ./03-perpexity.md ./04-srilm.md ./05-nnlm.md ./06-application.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/9.language-modeling.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-n-gram.md ./03-perpexity.md ./04-srilm.md ./05-nnlm.md ./06-application.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/9.language-modeling.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [신경망 기계번역](10-neural_machine_translation/00-cover.md)
#   * [서문](10-neural_machine_translation/01-intro.md)
#   * [Sequence-to-Sequence](10-neural_machine_translation/02-seq2seq.md)
#   * [Attention](10-neural_machine_translation/03-attention.md)
#   * [Input Feeding](10-neural_machine_translation/04-input-feeding.md)
#   * [Auto-regressive 속성과 Teacher Forcing](10-neural_machine_translation/05-teacher-forcing.md)
#   * [탐색(추론)](10-neural_machine_translation/06-beam-search.md)
#   * [성능 평가 방법](10-neural_machine_translation/07-eval.md)
#   * [정리](10-neural_machine_translation/08-conclusion.md)
echo '10-neural_machine_translation'
cd ./10-neural_machine_translation
time pandoc ./00-cover.md ./01-intro.md ./02-seq2seq.md ./03-attention.md ./04-input-feeding.md ./05-teacher-forcing.md ./06-beam-search.md ./07-eval.md ./08-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/10.nmt.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-seq2seq.md ./03-attention.md ./04-input-feeding.md ./05-teacher-forcing.md ./06-beam-search.md ./07-eval.md ./08-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/10.nmt.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [신경망 기계번역 심화 주제](11-adv_neural_machine_translation/00-cover.md)
#   * [다국어 번역](11-adv_neural_machine_translation/01-multilingual-nmt.md)
#   * [단방향 코퍼스를 활용하기](11-adv_neural_machine_translation/02-monolingual-corpus.md)
#   * [트랜스포머](11-adv_neural_machine_translation/03-transformer.md)
#   * [정리](11-adv_neural_machine_translation/04-conclusion.md)
echo '11-adv_neural_machine_translation'
cd ./11-adv_neural_machine_translation
time pandoc ./00-cover.md ./01-multilingual-nmt.md ./02-monolingual-corpus.md ./03-transformer.md ./04-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/11.adv-nmt.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-multilingual-nmt.md ./02-monolingual-corpus.md ./03-transformer.md ./04-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/11.adv-nmt.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [강화학습을 활용한 자연어생성](12-reinforcement_learning/00-cover.md)
#   * [서문](12-reinforcement_learning/01-intro.md)
#   * [강화학습 기초](12-reinforcement_learning/02-rl_basics.md)
#   * [폴리시 그래디언트](12-reinforcement_learning/03-policy-gradient.md)
#   * [자연어생성에서의 강화학습의 특성](12-reinforcement_learning/04-characteristic.md)
#   * [강화학습을 활용한 지도학습](12-reinforcement_learning/05-supervised-nmt.md)
#   * [강화학습을 활용한 비지도학습](12-reinforcement_learning/06-unsupervised-nmt.md)
#   * [정리](12-reinforcement_learning/07-conclusion.md)
echo '12-reinforcement_learning'
cd ./12-reinforcement_learning
time pandoc ./00-cover.md ./01-intro.md ./02-rl_basics.md ./03-policy-gradient.md ./04-characteristic.md ./05-supervised-nmt.md ./06-unsupervised-nmt.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/12.rl.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-rl_basics.md ./03-policy-gradient.md ./04-characteristic.md ./05-supervised-nmt.md ./06-unsupervised-nmt.md ./07-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/12.rl.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [듀얼리티 활용하기](13-duality/00-cover.md)
#   * [듀얼리티란](13-duality/01-intro.md)
#   * [듀얼리티를 활용한 지도학습](13-duality/02-dsl.md)
#   * [듀얼리티를 활용한 비지도학습](13-duality/03-dul.md)
#   * [쉬어가기: Back-translation을 재해석 하기](13-duality/04-back_translation.md)
#   * [정리](13-duality/05-conclusion.md)
echo '13-duality'
cd ./13-duality
time pandoc ./00-cover.md ./01-intro.md ./02-dsl.md ./03-dul.md ./04-back_translation.md ./05-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/13.duality.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-dsl.md ./03-dul.md ./04-back_translation.md ./05-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/13.duality.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [서비스 만들기](14-productization/00-cover.md)
#   * [파이프라인](14-productization/01-pipeline.md)
#   * [구글의 신경망 기계번역](14-productization/02-gnmt.md)
#   * [에딘버러 대학의 신경망 기계번역](14-productization/03-nematus.md)
#   * [마이크로소프트의 신경망 기계번역](14-productization/04-microsoft.md)
echo '14-productization'
cd ./14-productization
time pandoc ./00-cover.md ./01-pipeline.md ./02-gnmt.md ./03-nematus.md ./04-microsoft.md --latex-engine=xelatex -o ../${DIR_PATH}/14.productization.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-pipeline.md ./02-gnmt.md ./03-nematus.md ./04-microsoft.md --latex-engine=xelatex -o ../${DIR_PATH}/14.productization.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [전이학습 활용하기](15-transfer_learning/00-cover.md)
#   * [전이학습이란](15-transfer_learning/01-intro.md)
#   * [기존의 방법](15-transfer_learning/02-previous_work.md)
#   * [ELMo](15-transfer_learning/03-elmo.md)
#   * [BERT](15-transfer_learning/04-bert.md)
#   * [기계번역에 적용하기](15-transfer_learning/05-machine_translation.md)
#   * [정리](15-transfer_learning/06-conclusion.md)
echo '15-transfer_learning'
cd ./15-transfer_learning
time pandoc ./00-cover.md ./01-intro.md ./02-previous_work.md ./03-elmo.md ./04-bert.md ./05-machine_translation.md ./06-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/15.transfer_learning.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ./00-cover.md ./01-intro.md ./02-previous_work.md ./03-elmo.md ./04-bert.md ./05-machine_translation.md ./06-conclusion.md --latex-engine=xelatex -o ../${DIR_PATH}/15.transfer_learning.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

# * [이 책을 마치며](epilogue.md)
# * [References](references.md)
echo 'epilogue'
cd ./14-productization
time pandoc ../epilogue.md ../references.md --latex-engine=xelatex -o ../${DIR_PATH}/16.epilogue.docx --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
time pandoc ../epilogue.md ../references.md --latex-engine=xelatex -o ../${DIR_PATH}/16.epilogue.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE} -V geometry:margin=${MARGIN} -V linestretch=${LINE_SPACE}
cd ..

time gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=./export/merged.pdf ./export/0.preface.pdf ./export/1.introduction.pdf ./export/2.basic-math.pdf ./export/3.pytorch-intro.pdf ./export/4.preprocessing.pdf ./export/5.word-senses.pdf ./export/6.word-embedding.pdf ./export/7.sequential-modeling.pdf ./export/8.text-classification.pdf ./export/9.language-modeling.pdf ./export/10.nmt.pdf ./export/11.adv-nmt.pdf ./export/12.rl.pdf ./export/13.duality.pdf ./export/14.productization.pdf ./export/15.transfer_learning.pdf ./export/16.epilogue.pdf
