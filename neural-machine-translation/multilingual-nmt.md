# Multi-lingual Neural Machine Translation

기존의 번역 시스템이 Seq2seq를 필두로 어느정도 안정된 성능을 제공함에 따라서, 이를 활용한 여러가지 추가적인 연구주제가 생겨났습니다. 하나의 end2end model에서 여러쌍의 번역을 동시에 제공하는 multi-lingual NMT model이 그 주제중에 하나입니다.

## Zero-shot Learning

이 흥미로운 방식은 [\[Johnson at el.2016\]](https://arxiv.org/pdf/1611.04558.pdf)에서 제안되었습니다. 이 방식의 특징은 multi-lingual corpus를 하나의 모델에 훈련시키면 부가적으로 parallel corpus에 존재하지 않은 언어쌍도 번역이 가능하다는 것 입니다. 즉, 한번도 모델에게 데이터를 보여준 적이 없지만 처리할 수 있기 때문에, zero-shot learning이라는 이름이 붙었습니다. (이 이름은 꼭 machine translation이 아니더라도 다양한 분야에서 사용되는 어휘입니다.) 뿐만 아니라, low resourced parallel corpus의 경우에도 부가적인 효과를 발휘 합니다.

방법은 너무나도 간단합니다. 아래와 같이 기존 parallel corpus의 맨 앞에 artificial token을 삽입함으로써 완성됩니다. 삽입된 token에 따라서 target sentence의 language가 결정됩니다.

- Hello, how are you? $$ \rightarrow $$ Hola, ¿cómo estás?
- **\<2es\>** Hello, how are you? $$ \rightarrow $$ Hola, ¿cómo estás?

### 1. Many to One

![](/assets/nmt-zeroshot-1.png)

### 2. One to Many

![](/assets/nmt-zeroshot-2.png)

### 3. Many to Many

![](/assets/nmt-zeroshot-3.png)

### 4. Zero-shot Translation

![](/assets/nmt-zeroshot-4.png)

### 5. Mixing Languages

![](/assets/nmt-zeroshot-5.png)

### 6. Limitation

### 7. Applications



