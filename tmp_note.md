# Temporal Note for Markdown Equations

## Add one Smoothing

$$
\begin{aligned}
P(w_i|w_{<i}) \approx \frac{C(w_{<i},w_i)+1}{C(w_{<i})+V}
\end{aligned}
$$

$$
\begin{aligned}
P(w_i|w_{<i}) &\approx \frac{C(w_{<i},w_i)+k}{C(w_{<i})+kV} \\
&\approx \frac{C(w_{<i},w_i)+(m/V)}{C(w_{<i})+m}
\end{aligned}
$$

$$
P(w_i|w_{<i}) \approx \frac{C(w_{<i},w_i)+m P(w_i)}{C(w_{<i})+m}
$$

## Auto-regressive and Teacher Forcing on RNNLM

$$
Y=argmax_Y P(Y|X)=argmax_Y \prod_{i=1}^{n}{P(y_i|X,y_{<i})}
$$

$$
or
$$
$$
\begin{aligned}
y_i=&argmax_y{P(y|X,y_{<i})} \\
&\text{where }y_0=BOS
\end{aligned}
$$

$$
\hat{y}_t=argmax_y{P(y|X,y_{<t};\theta)\text{ where }X=\{x_1,x_2,\cdots,x_n\}\text{ and }Y=\{y_0,y_1,\cdots,y_{m+1}\}}
$$

$$
\begin{aligned}
\mathcal{L}(Y)&=-\sum_{i=1}^{m+1}{\log{P(y_i|X,y_{<i};\theta)}} \\
\theta &\leftarrow \theta-\lambda\frac{1}{N}\sum_{i=1}^{N}{\mathcal{L}(Y_i)}
\end{aligned}
$$