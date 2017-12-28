# Sequence to Sequence

## Architecture Overview

## ![](/assets/seq2seq_architecture.png)

먼저 번역 또는 seq2seq 모듈이 하는 작업을 간단하게 수식 화 해보겠습니다.

$$
P(Y|X)~where~X=\{x_1,x_2,\cdots,x_n\},~Y=\{y_1,y_2,\cdots,y_m\}
$$

즉, source 문장 $$ X $$를 받아서 target 문장 $$ Y $$를 생성 해 내는 작업을 하게 됩니다.

seq2seq는 크게 3 서브 모듈로 구성되어 있습니다. 

- Encoder: 인코더는 source 문장을 입력으로 받아 문장을 함축하는 의미의 vector로 만들어 냅니다. $$ P(X) $$를 구하는 작업을 수행한다고 볼 수 있습니다.
- Decoder
- Generator: 이 모듈은 Decoder에서 vector를 받아 softmax를 계산하는 단순한 작업을 하는 모듈 입니다. 

## Embedding Layer

### 설명

### 코드

## Encoder

### 설명

### 코드

## Decoder

### 설명

### 코드

## Generator

### 설명

### 코드



