# Naive Bayes

## Maximum A Posterior

$$
\underbrace{P(Y|X)}_{posterior}=\frac{\overbrace{P(X|Y)}^{likelihood}\overbrace{P(Y)}^{prior}}{\underbrace{P(X)}_{evidence}}
$$

$$
\hat{y}_{MAP}=argmax_{y\in\mathcal{Y}}P(Y=y|X)
$$

$$
\hat{y}_{MLE}=argmax_{y\in\mathcal{Y}}P(X|Y=y)
$$

## Naive Bayes

$$
\begin{aligned}
P(Y=c|X=w_1,w_2,\cdots,w_n) &\varpropto P(X=w_1,w_2,\cdots,w_n|Y=c)P(Y=c) \\
&\approx P(w_1|c)P(w_2|c)\cdots P(w_n|c)P(c) \\
&=\prod_{i=1}^{n}{P(w_i|c)}P(c)
\end{aligned}
$$

$$
\begin{aligned}
\hat{c} &= argmax_{c \in \mathcal{C}}{P(Y=c|X=w_1,w_2,\cdots,w_n)} \\
&=argmax_{c \in \mathcal{C}}{\prod_{i=1}^{n}{P(w_i|c)}P(c)}
\end{aligned}
$$

$$
\tilde{P}(Y=c)=\frac{Count(c)}{\sum_{i=1}^{|\mathcal{C}|}{Count(c_i)}}
$$

$$
\tilde{P}(w|c)=\frac{Count(w,c)}{\sum_{j=1}^{|V|}{Count(w_j,c)}}
$$

### Example: Sentiment Analysis

$$
\begin{gathered}
\mathcal{C}=\{\color{blue}positive\color{default},\color{red}negative\color{default}\} \\
\mathcal{D}=\{d_1,d_2,\cdots\}
\end{gathered}
$$

$$
\begin{aligned}
P(\color{blue}positive\color{default}|I,am,happy,to,see,this,movie)&= \frac{P(I,am,happy,to,see,this,movie|\color{blue}positive\color{default})P(\color{blue}positive\color{default})}{P(I,am,happy,to,see,this,movie)}\\
&\approx \frac{P(I|\color{blue}positive\color{default})P(am|\color{blue}positive\color{default})P(happy|\color{blue}positive\color{default})\cdots P(movie|\color{blue}positive\color{default})P(\color{blue}positive\color{default})}{P(I,am,happy,to,see,this,movie)} \\
\\
P(happy|\color{blue}positive\color{default})&\approx\frac{Count(happy, \color{blue}positive\color{default})}{\sum_{j=1}^{|V|}{Count(w_j,\color{blue}positive\color{default})}} \\
P(\color{blue}positive\color{default})&\approx\frac{Count(\color{blue}positive\color{default})}{|\mathcal{D}|}
\end{aligned}
$$

$$
\begin{aligned}
P(\color{red}negative\color{default}|I,am,happy,to,see,this,movie)&= \frac{P(I,am,happy,to,see,this,movie|\color{red}negative\color{default})P(\color{red}negative\color{default})}{P(I,am,happy,to,see,this,movie)}\\
&\approx \frac{P(I|\color{red}negative\color{default})P(am|\color{red}negative\color{default})P(happy|\color{red}negative\color{default})\cdots P(movie|\color{red}negative\color{default})P(\color{red}negative\color{default})}{P(I,am,happy,to,see,this,movie)} \\
\\
P(happy|\color{red}negative\color{default})&\approx\frac{Count(happy, \color{red}negative\color{default})}{\sum_{j=1}^{|V|}{Count(w_j,\color{red}negative\color{default})}} \\
P(\color{red}negative\color{default})&\approx\frac{Count(\color{red}negative\color{default})}{|\mathcal{D}|}
\end{aligned}
$$

$$
\begin{gathered}
P(\color{blue}positive\color{default}|I,am,not,happy,to,see,this,movie) \\
P(\color{red}negative\color{default}|I,am,not,happy,to,see,this,movie) \\
\\
P(not,happy) \neq P(not)P(happy)
\end{gathered}
$$

## Add-one Smoothing

$$
\begin{gathered}
P(happy|\color{blue}positive\color{default})\approx\frac{Count(happy, \color{blue}positive\color{default})}{\sum_{j=1}^{|V|}{Count(w_j,\color{blue}positive\color{default})}}=0, \\
\\
\text{where }Count(happy, \color{blue}positive\color{default})=0.
\end{gathered}
$$

$$
\tilde{P}(w|c)=\frac{Count(w,c)+1}{\big(\sum_{j=1}^{|V|}{Count(w_j,c)}\big)+|V|}
$$