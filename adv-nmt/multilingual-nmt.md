# Multi-lingual Neural Machine Translation

이제부터는 기계번역의 성능을 끌어올리기 위한 고급 기법들을 설명하고자 합니다. 코드를 직접 구현하고 실습을 해 보기보단, 논문을 소개하는 위주가 될 것 입니다. 앞으로 소개할 기술(논문)들은 일부는 기계번역에만 적용 가능한 기술들도 있지만, 자연어 생성 또는 시퀀스 데이터 생성에 응용될 수 있는 기술들도 있습니다.

기존의 번역 시스템이 Seq2seq를 필두로 어느정도 안정된 성능을 제공함에 따라서, 이를 활용한 여러가지 추가적인 연구 주제가 생겨났습니다. 하나의 end2end 모델에서 여러 언어쌍의 번역을 동시에 제공하는 multi-lingual 기계번역이 그 주제중에 하나입니다.

## Zero-shot Learning

이 흥미로운 방식은 [[Johnson at el.2016]](https://arxiv.org/pdf/1611.04558.pdf)에서 제안 되었습니다. 이 방식의 특징은 여러 언어쌍의 parallel 코퍼스를 하나의 모델에 훈련하면 부가적으로 parallel 코퍼스에 존재하지 않은 언어쌍도 번역이 가능하다는 것 입니다. 즉, 한번도 기계번역 모델에게 데이터를 보여준 적이 없지만, 해당 언어쌍 번역을 처리할 수 있기 때문에, zero-shot learning이라는 이름이 붙었습니다. 뿐만 아니라, parallel 코퍼스가 적은 경우에도 부가적인 효과를 발휘 합니다.

방법은 너무 간단합니다. 아래와 같이 기존 parallel 코퍼스의 맨 앞에 특수 토큰을 삽입함으로써 완성됩니다. 삽입된 토큰에 따라서 target 언어가 결정됩니다.

- Hello, how are you? $\rightarrow$ Hola, ¿cómo estás?
- $<2\text{es}>$ Hello, how are you? $\rightarrow$ Hola, ¿cómo estás?

실험의 목표는 단순히 Multi-lingual end2end model을 구현하는 것이 아닌, 다른 언어쌍의 코퍼스를 활용하여 특정 언어쌍 번역기의 성능을 올릴 수 있는가에 대한 관점도 있습니다. 이에 따라 실험은 크게 4가지 관점에서 수행되었습니다.

|번호|방법|설명|
|-|-|-|
|1|Many to One|다수의 언어를 encoder에 넣고 훈련시킵니다.|
|2|One to Many|다수의 언어를 decoder에 넣고 훈련시킵니다.|
|3|Many to Many|다수의 언어를 encoder와 decoder에 모두 넣고 훈련시킵니다.|
|4|Zero-shot Translation|위의 방법으로 훈련된 모델에서 zero-shot translation의 성능을 평가합니다.|

언어가 다른 코퍼스를 하나로 합치다보면 양이 다르기 때문에 이에 대한 대처 방법도 정의 되어야 합니다. 따라서 아래의 실험들에서 oversampling 기법의 사용 유무도 같이 실험이 되었습니다. Oversampling 기법은 양이 적은 코퍼스를 양이 많은 코퍼스에 양과 비슷하도록 (데이터를 반복시켜) 양을 늘려 맞춰주는 방법을 말합니다.

### Many to One

![Many to One](../assets/nmt-zeroshot-1.png)

이 실험에서는 전체적으로 성능이 향상 된 것을 볼 수 있습니다. 하단의 일본어, 한국어, 스페인어, 포르투갈어 실험의 경우에는 모두 oversampling을 기준으로 실험되었습니다.

### One to Many

![One to Many](../assets/nmt-zeroshot-2.png)

이 실험에서는 이전 실험과 달리 성능의 향상이 있다고 보기 힘듭니다. 게다가 oversampling과 관련해서 코퍼스의 양이 적은 영어/독일어 코퍼스는 oversampling의 이득을 본 반면, 양이 충분한 영어/프랑스어 코퍼스의 경우에는 oversampling을 하면 더 큰 손해를 보는 것을 볼 수 있습니다.

### Many to Many

![Many to Many](../assets/nmt-zeroshot-3.png)

이 실험에서도 대부분의 실험결과가 성능의 하락으로 이어졌습니다. (그렇지만 절대적인 BLEU 수치는 쓸만합니다.)

### Zero-shot Translation

![Zero-shot Translation](../assets/nmt-zeroshot-4.png)

이 실험은 Zero-shot learning의 성능을 평가하였습니다. bridged 방법은 중간 언어를 영어로 하여 Portuguese $\rightarrow$ English $\rightarrow$ Spanish 2단계에 걸쳐 번역을 한 경우를 말합니다. (PBMT 방식은 SMT방식 중의 하나입니다.) NMT Pt $\rightarrow$ Es는 단순 parallel 코퍼스를 활용하여 기존의 방법대로 훈련한 baseline입니다.

모델1은 Pt $\rightarrow$ En, En $\rightarrow$ Es를 한 모델에 훈련한 버전 입니다. 그리고 모델2는 En $\leftrightarrow$ Pt, En $\leftrightarrow$ Es 코퍼스를 한 모델에 훈련한 버전 입니다. 모델2는 총 4가지 코퍼스를 훈련 한 점을 주의해야 합니다.

마지막으로 모델2 + incremental 학습 방식은 $(c)$ 보다 적은양의 parallel 코퍼스를 기존에 훈련한 모델2에 추가적으로 훈련한 모델입니다.

비록 모델1과 모델l2는 훈련 중에 한번도 Pt $\rightarrow$ Es 데이터를 보지 못했지만, 20이 넘는 BLEU를 보여주는 것을 알 수 있습니다. 하지만 bridge 방식의 $(a),(b)$ 보다 성능이 떨어지는 것을 알 수 있습니다. 다행히도 $(f)$의 경우에는 $(c)$보다 (큰 차이는 아니지만) 성능이 뛰어난 것을 알 수 있습니다. 따라서 우리는 parallel 코퍼스의 양이 얼마 되지 않는 언어쌍의 번역기를 훈련할 때에 위와 같은 방법을 통해서 성능을 끌어올릴 수 있음을 알 수 있습니다.

### Conclusion

앞서 다룬 monlingual 코퍼스를 활용하는 방법의 연장선상으로써 위와 같이 multilingual 기계번역 모델로서는 의의가 있지만, 성능에 있어서는 이득이 있었기 때문에 실제 사용에는 한계가 있습니다. 더군다나 훈련 데이터의 양이 적은 언어쌍에 대해서는 성능의 향상이 있지만 뒤 챕터에 설명할 방법들을 사용하면 그다지 좋은 방법은 아닙니다.

### Applications

우리는 합성 토큰을 추가하는 방식을 다른 곳에서도 응용할 수 있습니다. 다른 도메인의 데이터를 하나로 모아 번역기를 훈련시키는 과정에 사용 가능합니다. 예를 들어 코퍼스를 뉴스기사와 미드 자막에서 각각 모았다고 가정하면, 문어체와 대화체로 도메인을 나누어 특수 토큰을 추가하여 우리가 원하는대로 번역문의 말투를 바꾸어줄 수 있을 겁니다. 또는 마찬가지로 의료용과 법률용으로 나누어 번역기의 모드를 바꾸어줄 수 있을 겁니다.
