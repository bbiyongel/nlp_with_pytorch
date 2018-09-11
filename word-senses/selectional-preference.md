# Selectional Preference

## Selectional Preference Strength

$$
\begin{aligned}
S_R(w)&=\text{KL}(P(C|w)||P(C)) \\
&=-\sum_{c\in\mathcal{C}}P(c|w)\log{\frac{P(c)}{P(c|w)}}
\end{aligned}
$$

## Selectional Association

$$
A_R(w,c)=-\frac{P(c|w)\log{\frac{P(c)}{P(c|w)}}}{S_R(w)}
$$

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