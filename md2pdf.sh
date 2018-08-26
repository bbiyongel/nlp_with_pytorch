FONT_SIZE=12pt

# * [소개글](README.md)
# * [Index](index_list.md)
echo 0
cd ./nlp-with-deeplearning
pandoc ../README.md ../index_list.md --latex-engine=xelatex -o ../0.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Temporal Note](tmp_note.md)
# * [Natural Language Processing with Deeplearning](nlp-with-deeplearning/cover.md)
#   * [Intro](nlp-with-deeplearning/intro.md)
#   * [Deeplearning](nlp-with-deeplearning/deeplearning.md)
#   * [Why NLP is difficult](nlp-with-deeplearning/why-nlp-difficult.md)
#   * [Why Korean NLP is Hell](nlp-with-deeplearning/korean-is-hell.md)
#   * [Recent Trends](nlp-with-deeplearning/trends.md)
echo 1
cd ./nlp-with-deeplearning
pandoc ./cover.md ./intro.md ./deeplearning.md ./why-nlp-difficult.md ./korean-is-hell.md ./trends.md --latex-engine=xelatex -o ../1.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Hello PyTorch](pytorch-intro/cover.md)
#   * [Intro](pytorch-intro/intro.md)
#   * [How to install](pytorch-intro/how-to-install.md)
#   * [PyTorch tutorial](pytorch-intro/hello-pytorch.md)
echo 2
cd ./pytorch-intro
pandoc ./cover.md ./intro.md ./how-to-install.md ./hello-pytorch.md --latex-engine=xelatex -o ../2.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Word Sense Disambiguation](wsd/cover.md)
#   * [Intro](wsd/intro.md)
#   * [Using Thesaurus](wsd/wordnet.md)
#   * [Using Feature Vector](wsd/feature-vector.md)
#   * [Selectional Preference](wsd/selectional-preference.md)
echo 3
cd ./wsd
pandoc ./cover.md ./wordnet.md ./feature-vector.md ./selectional-preference.md --latex-engine=xelatex -o ../3.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Preprocessing](preprocessing/cover.md)
#   * [Intro](preprocessing/intro.md)
#   * [Collecting corpus](preprocessing/collecting-corpus.md)
#   * [Cleaning corpus](preprocessing/cleaning-corpus.md)
#   * [Tokenization \(POS Tagging\)](preprocessing/tokenization.md)
#   * [Aligning parallel corpus](preprocessing/align.md)
#   * [Subword Segmentation](preprocessing/bpe.md)
#   * [Detokenization](preprocessing/detokenization.md)
#   * [TorchText](preprocessing/torchtext.md)
echo 4
cd ./preprocessing
pandoc ./cover.md ./intro.md ./collecting-corpus.md ./cleaning-corpus.md ./tokenization.md ./align.md ./bpe.md ./detokenization.md ./torchtext.md --latex-engine=xelatex -o ../4.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Word Embedding Vector](word-embedding-vector/cover.md)
#   * [Intro](word-embedding-vector/intro.md)
#   * [One-hot encoding](word-embedding-vector/one-hot-encoding.md)
#   * [Previous methods](word-embedding-vector/previous-methods.md)
#   * [Word2Vec](word-embedding-vector/word2vec.md)
#   * [GloVe](word-embedding-vector/glove.md)
#   * [FastText](word-embedding-vector/fasttext.md)
#   * [Doc2Vec](word-embedding-vector/doc2vec.md)
echo 5
cd ./word-embedding-vector
pandoc ./cover.md ./intro.md ./one-hot-encoding.md ./previous-methods.md ./word2vec.md ./glove.md ./fasttext.md ./doc2vec.md --latex-engine=xelatex -o ../5.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Sequence Modeling](sequential-modeling/cover.md)
#   * [Intro](sequential-modeling/intro.md)
#   * [Recurrent Neural Network](sequential-modeling/rnn.md)
#   * [Long Short Term Memory](sequential-modeling/lstm.md)
#   * [Gated Recurrent Unit](sequential-modeling/gru.md)
#   * [Gradient Clipping](sequential-modeling/gradient-clipping.md)
echo 6
cd ./sequential-modeling
pandoc ./cover.md ./intro.md ./rnn.md ./lstm.md ./gru.md ./gradient-clipping.md --latex-engine=xelatex -o ../6.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Text Classification](text-classification/cover.md)
#   * [Intro](text-classification/intro.md)
#   * [Naive Bayes](text-classification/naive-bayes.md)
#   * [Using CNN](text-classification/cnn.md)
#   * [Using RNN](text-classification/rnn.md)
#   * [Unsupervised Text Classification](text-classification/unsupervised-text-classification.md)
echo 7
cd ./text-classification
pandoc ./cover.md ./intro.md ./naive-bayes.md ./cnn.md ./rnn.md ./unsupervised-text-classification.md --latex-engine=xelatex -o ../7.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Language Modeling](language-modeling/cover.md)
#   * [Intro](language-modeling/intro.md)
#   * [n-gram](language-modeling/n-gram.md)
#   * [Perpexity](language-modeling/perpexity.md)
#   * [n-gram Exercise](language-modeling/srilm.md)
#   * [Neural Network Language Model](language-modeling/nnlm.md)
#   * [Applications](language-modeling/application.md)
echo 8
cd ./language-modeling
pandoc ./cover.md ./intro.md ./n-gram.md ./perpexity.md ./srilm.md ./nnlm.md ./application.md --latex-engine=xelatex -o ../8.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Neural Machine Translation](neural-machine-translation/cover.md)
#   * [Intro](neural-machine-translation/intro.md)
#   * [Sequence-to-Sequence](neural-machine-translation/seq2seq.md)
#   * [Attention](neural-machine-translation/attention.md)
#   * [Input Feeding](neural-machine-translation/input-feeding.md)
#   * [Auto-regressive and Teacher Forcing](neural-machine-translation/teacher-forcing.md)
#   * [Search](neural-machine-translation/beam-search.md)
#   * [Evaluation](neural-machine-translation/eval.md)
#   * [Source Code](neural-machine-translation/code.md)
echo 9
cd ./neural-machine-translation
pandoc ./cover.md ./intro.md ./seq2seq.md ./attention.md ./input-feeding.md ./teacher-forcing.md beam-search.md eval.md code.md --latex-engine=xelatex -o ../9.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Advanced Topic on NMT](adv-nmt/cover.md)
#   * [Multilingual NMT](adv-nmt/multilingual-nmt.md)
#   * [Using Monolingual Corpora](adv-nmt/monolingual-corpus.md)
#   * [Fully Convolutional Seq2seq](adv-nmt/fconv.md)
#   * [Transformer](adv-nmt/transformer.md)
echo 10
cd ./adv-nmt
pandoc ./cover.md ./multilingual-nmt.md ./monolingual-corpus.md ./fconv.md transformer.md --latex-engine=xelatex -o ../10.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=12pt
cd ..
# * [NLP with Reinforcement Learning](reinforcement-learning/cover.md)
#   * [Intro](reinforcement-learning/intro.md)
#   * [Policy Gradients](reinforcement-learning/policy-gradient.md)
#   * [RL on NLG](reinforcement-learning/characteristic.md)
#   * [Supervised NMT](reinforcement-learning/supervised-nmt.md)
#   * [Unsupervised NMT](reinforcement-learning/unsupervised-nmt.md)
echo 11
cd ./reinforcement-learning
pandoc ./cover.md ./intro.md ./policy-gradient.md ./characteristic.md ./supervised-nmt.md ./unsupervised-nmt.md --latex-engine=xelatex -o ../11.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Exploit Duality](duality/cover.md)
#   * [Duality](duality/intro.md)
#   * [Dual Supervised Learning](duality/dsl.md)
#   * [Dual Unsupervised Learning](duality/dul.md)
echo 12
cd ./duality
pandoc ./cover.md ./intro.md ./dsl.md ./dul.md --latex-engine=xelatex -o ../12.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [Productization](productization/cover.md)
#   * [Pipeline](productization/pipeline.md)
#   * [Google's NMT](productization/gnmt.md)
#   * [Edinburgh's NMT](productization/nematus.md)
#   * [Booking.com's NMT](productization/booking-com.md)
#   * [Microsoft's NMT](productization/microsoft.md)
echo 13
cd ./productization
pandoc ./cover.md ./pipeline.md ./gnmt.md ./nematus.md ./booking-com.md ./microsoft.md --latex-engine=xelatex -o ../13.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
cd ..
# * [References](references.md)
echo 14
pandoc references.md --latex-engine=xelatex -o ./14.pdf --variable mainfont='Nanum Myeongjo' -V fontsize=${FONT_SIZE}
