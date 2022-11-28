# Bayesian-Analysis-on-Investor-Behaviors
 
### Introduction
In a short word, *I want to know if the stock investors change change their behaviors in different periods*.

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
#### 1. Assumptions and Theory
The initial idea is that $𝑝𝑟𝑜𝑏_{𝑑𝑖𝑓𝑓}(𝑑𝑖,𝑡)$ distribute in normal form. 
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

$$p(\lambda_{1},\lambda_{2},M|y_{1},\dots,y_{n})\propto p(\lambda_{1},\lambda_{2},M)p(y_{1},\dots,y_{n}|\lambda_{1},\lambda_{2},M)\propto \lambda_{1}^{a_{1}+\sum_{t=1}^M y_{t}-1} \times e^{-\lambda_{1}(b_{1}+M)}\times \lambda_{2}^{a_{2}+\sum_{t=M+1}^{n}y_{t}-1} \times e^{-\lambda_{2}(b_{2}+n-M)}$$

Do Gibbs sampling for posterior distribution $𝑝\lambda_{1},\lambda_{2},𝑀|𝑦_{1},\dots,𝑦_{𝑛}$:
$$p(\lambda+{1}|\dots)\sim Gamma(a_{1}+\sum_{t=1}^M y_{t} ,b_{1}+M)$$
$$p(\lambda+{2}|\dots)\sim Gamma(a_{2}+\sum_{t=M+1}^M y_{t} ,b_{2}+M)$$

#### 2.	Initialization
For each group $G_{d_{i}}$, we assume $\mu=mean(Y_{t})$, $\sigma=sd(Y_{t})$, and we select values for a,b to fit them. 
We assume $a_{1}=a_{2}=a=\frac{\mu^2}{\sigma^2}$ , $b_{1}=b_{2}=b=\frac{\mu}{\sigma^2}$.
The starting value of $\lambda_{1}=\mu$, $\lambda_{2}=\mu$

For example, in $G_{20},1prob$ ($G_{20}$ represents the group of invester with local extreme equals to 20, which we can consider as longer time stock holders) case, $n=557$, $\mu=0.126$, $\sigma=0.106$, the distribution of prior $p(\lambda_{1})$ is like:

![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/2_1_1.png)

#### 3. Gibbs Sampling and MCMC

Do the sampling for $1000$ times. 
In each iteration, we resample $\lambda_{1}$  ,$\lambda_{2}$ from Gamma distribution $Gamma(a_{1}+lowersum_{m},b_{1}+m)$ and $Gamma(a_{2}+uppersum_{m},b_{2}+n-m)$. 
We calculate $b^{new}=((\log⁡(\lambda_{1})+\log⁡(\lambda_{2})))/2\times \sum^n y_{t} +(\lambda_{2}-\lambda_{1})\times n/2$.
After Gibbs sampling, we set first 100 times as burn-in sampling and do MCMC.

### Summary
In $G_{20},1prob$ case, the posterior densities of $\lambda_{1}$ and $\lambda_{2}$ are as below:

![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/2_4_1.png)

PMF of Change Point is as below:

![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/2_4_2.png)

We notice that data indexes around 340(2020/07/24) to 430(2021/04/20) have some high probabilities. 
During this period, the US government announced the second and third Economic Impact Payments, 
which were considered as possible factor that changed the investors behaviors.

If we plot the point estimates as below, we can see that after date index of 400, the probabilities have a higher mean.

![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/2_4_3.png)

### Discussion and Implications
So in $G_{20},1prob$ case, it looks like the parameter $\lambda$ increase after around September 2020. 
If this is true, it means the average of probabilities that “QQQ ma3 price will decrease tomorrow” increase in the second period. 
This does not directly mean QQQ price is more likely to crash after September 2020. 
It may indicate that stocks are more likely to shape “significant” down signals, which usually result from more significant fluctuations, which result from more active investor behaviors.

In $G_{20},2prob$ case, the two $\lambda$ do not change much. 
Maybe $\lambda_{2}$ is slightly smaller than $\lambda_{1}$, 
but there is no significant date point that has high PMF of change point. 
Therefore, the model does not give us great confidence that $G_{20},2prob$ has change its distribution along times.

![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/2_5_1.png)

![image](https://github.com/highwaycus/Bayesian-Analysis-on-Investor-Behaviors/blob/main/2_5_2.png)

What intuition can we get from combining the results of $G_{20},1prob$ and $G_{20},2prob$? 
Group $G_{20}$ represents the behaviors of relatively long-term stockholders, 
we may conclude that after the shock and mass in stock market in 2020, 
these investors are more sensitive (or active) when meet a possible price-decreasing event, 
while they still maintain their patience when a boost signal is coming. 

Now let’s see what happen to other case. For Group $G_{3}$ and $G_{5}$, 
which represents the group of investors who tend to hold stock for only short time, does not appear significant difference between $\lambda_{1}$ and $\lambda_{2}$. 

For Group $G_{8}$ and $G_{13}$, it has similar results with Group $G_{20}$: 
the distribution of 1_prob change ($\lambda_{2}\gtr\lambda_{1}$) while the 2_prob does not show significant change in distribution parameters. 
And the change points for 1_prob are also similar to what we obtained in $G_{20}$ experiment.

From the results of our experiments, we can conclude that the possible reason that K chart model lost its prediction power after September 2020 is the change of behaviors of the long-term investors toward price-decreasing events. 
They become more conservative (or say, sensitive) toward the events that may cause to decreasing in price. Or we can say, they become more risk-averse toward possible loss.
