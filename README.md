# Bayesian-Analysis-on-Investor-Behaviors

### Introduction
In a short word, I want to know if the stock investors change change their behaviors in different periods.

In my previous project, k-chart indicator for etf1market, Our team designed a model (k-chart model) to estimate the probabilities of price goes up and down for a stock.
By observing the past data, I can compute a criterion to determine the prediction. 
For example, if the model output is (0.3, 0.2, 0.5), which represents probabilities of down, up, not clear (remain), and the prediction is
$$𝑝𝑟𝑖𝑐𝑒\quad 𝑢𝑝\quad 𝑖𝑓\quad Pr(𝑑𝑜𝑤𝑛)−Pr(𝑢𝑝)<0.1$$ 
$$𝑝𝑟𝑖𝑐𝑒\quad 𝑑𝑜𝑤𝑛\quad 𝑖𝑓\quad Pr(𝑑𝑜𝑤𝑛)−Pr(𝑢𝑝)≥0.1$$
The model remained 0.7 accurate rate since 2015. 
However, after the eruption of Covid-19, we observe that the algorithm become more “inaccurate”. 
More specifically, it looks like the weights for the different investor groups have changed. 
I want to use Bayesian models to capture the change (if it exists). 
If we can show the change exist, then we may get some insights about how to adjust the k-chart algorithm.

To verify the hypothesis, I want to use Gibbs sampling. 
We will observe if the parameters really change in the two periods. 
And this problem would be more complicated because it can be regarded as multivariate distribution.

### Data source
Stcok and ETF price data source: Yahoo Finance. 
The original data source is load from Python Yahoo Finance API, it’s a free source. The example code is as below
```{python}
import yfinance as yf
dt = yf.Ticker("AAPL")
print(dt.info)
hist = dt.history(period="5d")
```

### The description of data
The date range is 2016/01/04 to 2021/10/29. Target ETF is QQQ, and several variables descriptions are as below:

|var |Description |1st Qu. |Median |3rd Qu.| Mean|SD|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|price |Daily close price of QQQ|136.43| 173.32| 241.32| 197.01| 82.20|
|N |Number of components (stock)|97| 98| 99| 97.81| 1.83|

### Modeling
#### Assumptions and Theory
The initial idea is that $𝑝𝑟𝑜𝑏_𝑑𝑖𝑓𝑓(𝑑𝑖,𝑡)$ distribute in normal form. 
But after observing the data (as figure below), I notice that the distributions skewed right, which makes them closer to poisson distribution than normal distribution. Therefore, I think I can use poisson model to represent these distributions, with appropriate parameter values $\lambda$.
![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/1_4_3.png)

For each group $G_{d_i}$, let $T_{d_i}$ be the trading date in which the distribution of $Y$ change. $Y$ is $1prob_{di}$ or $2prob_{di}$. 
Notice that we will change value of date from “year+month+date” to ordinal interger.

$$M\in\{1,2,\dots,n-1\}$$
$$y_{d_{i}},t|\lambda_{1},M\sim Poisson(\lambda_{1}),t=1,\dots,M$$
$$y_{d_{i}},t|\lambda_{2},M\sim Poisson(\lambda_{2}),t=M+1,\dots,n$$

The conjugate priors of $\lambda_{1}$ and $\lambda_{2}$ are

$$\lambda_{1}\sim 𝐺𝑎𝑚𝑚𝑎(𝑎_{1},𝑏_{1})$$
$$\lambda_{2}\sim 𝐺𝑎𝑚𝑚𝑎(𝑎_{2},𝑏_{2})$$

For $M$ we use a uniform distribution over the set $\{1, 2, .., n-1\}$,
$$M\sim Discrete(\{1,2,\dots,(n-1)\},）,(p_{1},p_{2},\dots,p_{(n-1)} ))$$
$$p_{m}=\frac{1}{n-1}\quad for\quad m=1,\dots,n-1$$

For posterior distribution, we know that

$$p(\lambda_{1},\lambda_{2},M|y_{1},\dots,y_{n})\propto p(\lambda_{1},\lambda_{2},M)p(y_{1},\dots,y_{n}|\lambda_{1},\lambda_{2},M)\propto \lambda_{1}^{a_{1}+\sum_{t=1}^{M}y_{t}-1} \times e^{-\lambda_{1}(b_{1}+M)}\times \lambda_{2}^{a_{2}+\sum_{t=M+1}^{n}y_{t}-1} \times e^{-\lambda_{2}(b_{2}+n-M)}$$

