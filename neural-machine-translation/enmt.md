# University of Edinburgh’s Neural MT Systems

사실 Google의 논문은 훌륭하지만 매우 scale이 큽니다. 저는 그래서 작은 scale의 기계번역 시스템에 관한 논문은 이 논문[\[Sennrich at el.2017\]](https://arxiv.org/pdf/1708.00726.pdf)을 높게 평가합니다. 이 논문도 기계번역 시스템을 구성할 때에 훌륭한 baseline이 될 수 있습니다.

## Subword Segmentation

![](/assets/nmt-edinburgh-1.png)
[[Sennrich at el.2016]](http://www.aclweb.org/anthology/P16-1162)

이 논문 또한 (그들이 처음으로 제안한 방식이기에) BPE 방식을 사용하여 tokenization을 수행하였습니다. 이쯤 되면 이제 subword 기반의 tokenization 방식이 정석이 되었음을 알 수 있습니다. 위의 code는 BPE algorithm에 대해서 간략하게 소개한 code 입니다.

## Architecture

## Monolingual Data

### Synthetic Data

## Ensemble