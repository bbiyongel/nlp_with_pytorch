# Align Parallel Corpus

대부분의 parallel corpora는 여러 문장 단위로 align이 되어 있는 경우가 많습니다. 이러한 경우에는 한 문장씩에 대해서 align을 해주어야 합니다. 또한, 이러한 과정에서 일부 parallel 하지 않은 문장들을 걸러내야 하고, 문장 간 align이 잘 맞지 않는 경우 align을 재정비 해 주거나 아예 걸러내야 합니다. 이러한 과정에 대해서 살펴 봅니다.

## Process Overview for Parallel Corpus Alignment

1. Building naive translator. (You may use pre-trained translator)
    1. Collect and normalize (clean + tokenize) corpus for each language.
    1. Get word embedding vector for each language.
    1. Get word-level-translator using [MUSE](https://github.com/facebookresearch/MUSE).
1. Align collected semi-parallel corpus based on naive translation using [Bleualign](https://github.com/rsennrich/Bleualign).
    1. Sentence-tokenize for each language.
    1. Normalize (clean + tokenize) corpus for each language.
    1. Get pseudo translation for each language from naive translator.
    1. Align parallel corpus using Bleualign.

## Building Naive Translator

## Align via Naive Translator