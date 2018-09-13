# Selectional Preference

이번 섹션에서는 selectional preference라는 개념에 대해서 다루어 보도록 하겠습니다. 문장은 여러 단어의 시퀀스로 이루어져 있습니다. 따라서 각 단어들은 문장내 주변의 단어들에 따라 그 의미가 정해지기 마련입니다. Selectional preference는 이를 좀 더 수치화하여 나타내 줍니다. 예를 들어 '마시다'라는 동사에 대한 목적어는 '<음료>' 클래스에 속하는 단어가 올 확률이 매우 높습니다. 따라서 우리는 '차'라는 단어가 '<탈 것>' 클래스에 속하는지 '<음료>' 클래스에 속하는지 쉽게 알 수 있습니다. 이런 성질을 이용하여 우리는 단어 중의성 해소(WSD)도 해결 할 수 있으며, 여러가지 문제들(syntactic disambiguation, semantic role labeling)을 수행할 수 있습니다.

## Selectional Preference Strength

앞서 언급한 것처럼 selectional preference는 단어와 단어 사이의 관계(예: verb-object)가 좀 더 특별한 경우에 대해 수치화 하여 나타냅니다. 술어(predicate) 동사(예: verb)가 주어졌을 때, 목적어(예: object)관계에 있는 단어(보통은 명사가 될 겁니다.)들의 분포는, 평소 문서 내에 해당 명사(예: object로써 noun)가 나올 분포와 다를 것 입니다. 그 분포의 차이가 크면 클수록 해당 술어(predicate)는 더 강력한 selectional preference를 갖는다고 할 수 있습니다. 이것을 Philip Resnik은 [[Resnik et al.1997](http://www.aclweb.org/anthology/W97-0209)]에서 Selectional Preference Strength라고 명명하고 KL-divergence를 사용하여 정의하였습니다.

$$
\begin{aligned}
S_R(w)&=\text{KL}(P(C|w)||P(C)) \\
&=-\sum_{c\in\mathcal{C}}P(c|w)\log{\frac{P(c)}{P(c|w)}} \\
&=-\mathbb{E}_{C\sim P(C|w)}[\log{\frac{P(C)}{P(C|W=w)}}]
\end{aligned}
$$

위의 수식을 해석하면, selectional preference strength $S_R(w)$은 $w$가 주어졌을 때의 object class $C$의 분포 $P(C|w)$와 그냥 해당 class들의 prior(사전) 분포 $P(C)$와의 KL-divergence로 정의되어 있음을 알 수 있습니다. 즉, selectional preference strength는 술어(predicate)가 특정 클래스를 얼마나 선택적으로 선호(selectional preference)하는지에 대한 수치라고 할 수 있습니다.

![클래스의 사전 확률 분포와 술어가 주어졌을 때의 확률 분포 변화](../assets/wsd-selectional-preference-strength.png)

예를 들어 '<음식>' 클래스의 단어는 '<공구>' 클래스의 단어보다 나타날 확률이 훨씬 높을 것 입니다. 이때, '사용하다'라는 동사(verb) 술어(predicate)가 주어진다면, 동사-목적어(verb-object) 관계에 있는 단어로써의 '<음식>' 클래스의 확률은 '<공구>' 클래스의 확률보다 낮아질 것 입니다.

## Selectional Association

이제 그럼 술어와 특정 클래스 사이의 선택 관련도를 어떻게 나타내는지 살펴보겠습니다. Selectional Association, $A_R(w,c)$은 아래와 같이 표현됩니다.

$$
A_R(w,c)=-\frac{P(c|w)\log{\frac{P(c)}{P(c|w)}}}{S_R(w)}
$$

위의 수식에 따르면, selectional preference strength가 낮은 술어(predicate)에 대해서 윗변의 값이 클 경우에는 술어와 클래스 사이에 더 큰 selectional association(선택 관련도)를 갖는다고 정의 합니다. 즉, selectional preference strength가 낮아서, 해당 술어(predicate)는 클래스(class)에 대한 선택적 선호 강도가 낮음에도 불구하고, 특정 클래스만 유독 술어에 영향을 받아서 윗변이 커질수록 selectional association의 수치도 커집니다.

예를 들어 어떤 아주 일반적인 동사에 대해서는 대부분의 클래스들이 prior(사전)확률 분포와 비슷하게 여전히 나타날 것입니다. 따라서 selectional preference strength, $S_R(w)$는 0에 가까울것 입니다. 하지만 그 가운데 해당 동사와 붙어서 좀 더 나타나는 클래스의 목적어가 있다면, selectional association $A_R(w,c)$는 매우 높게 나타날 것 입니다.

## Selectional Preference and WSD

## Similarity-based Selectional Preference

$$
(w,v,R),\text{ where }R\text{ is a relationship, such as verb-object}.
$$

$$
A_R(w,v_0)=\sum_{v\in\text{Seen}_R(w)}{\text{sim}(v_0,v)\cdot \phi_R(w,v)}
$$

$$
\phi_R(w,v)=\text{IDF}(v)
$$

[[Erk et al.2007](http://www.aclweb.org/anthology/P07-1028)]

## Pseudo Word

[[Chambers et al.2010](https://web.stanford.edu/~jurafsky/chambers-acl2010-pseudowords.pdf)]

## Selectional Preference Evaluation using Pseudo Word

## Example