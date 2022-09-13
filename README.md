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
