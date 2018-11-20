# Monty-hall Problem

$$
\begin{aligned}
P(C=2|A=0,B=1)&=\frac{P(A=0,B=1,C=2)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=2)P(A=0,C=2)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=2)P(A=0)P(C=2)}{P(B=1|A=0)P(A=0)} \\
&=\frac{1 \times \frac{1}{3}}{\frac{1}{2}}=\frac{2}{3},\\
\text{where }P(B=1|A=0)=&\frac{1}{2},~P(C=2)=\frac{1}{3},\text{ and }P(B=1|A=0,C=2)=1.
\end{aligned}
$$

$$
\begin{aligned}
P(C=0|A=0,B=1)&=\frac{P(A=0,B=1,C=0)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=0)P(A=0,C=0)}{P(A=0,B=1)} \\
&=\frac{P(B=1|A=0,C=0)P(A=0)P(C=0)}{P(B=1|A=0)P(A=0)} \\
&=\frac{\frac{1}{2} \times \frac{1}{3}}{\frac{1}{2}}=\frac{1}{3},\\
\text{where }&P(B=1|A=0,C=0)=\frac{1}{2}
\end{aligned}
$$