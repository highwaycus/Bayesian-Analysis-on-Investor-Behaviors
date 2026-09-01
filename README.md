# Bayesian Changepoint Detection in Investor Behavior

This project uses Bayesian changepoint detection to find shifts in daily QQQ transaction activity. A custom Gibbs sampler models the counts as a piecewise Poisson process and estimates where the underlying transaction rate changes.

## What I wanted to understand

Investor activity does not stay at one level forever. I wanted to identify the points where the behavior appears to change instead of forcing one model to describe the entire time series.

## Model

Let the observed daily transaction counts be (y_1, ldots, y_T). The model assumes one changepoint (	au):

[
y_t sim
egin{cases}
	ext{Poisson}(lambda_1), & t le 	au \
	ext{Poisson}(lambda_2), & t > 	au
end{cases}
]

The Poisson rates use Gamma priors, and the changepoint uses a uniform prior:

[
lambda_1, lambda_2 sim 	ext{Gamma}(alpha, eta),
qquad
	au sim 	ext{Uniform}(1, T)
]

## Implementation

The Gibbs sampler iteratively updates:

1. (lambda_1), using observations before the changepoint
2. (lambda_2), using observations after the changepoint
3. (	au), using its conditional posterior distribution

The implementation also includes:

- Trace plots for convergence checks
- Posterior summaries of the changepoint
- Visualization of the transaction series with estimated change locations

## Current output

The model produces posterior estimates for changepoints in the QQQ transaction-count series. Connecting those changes to specific market events still requires separate validation.

## Next steps

- Extend the model to multiple changepoints
- Compare the result with frequentist methods
- Test alternative count models, such as the negative binomial
- Connect detected changes with market events and sentiment data
