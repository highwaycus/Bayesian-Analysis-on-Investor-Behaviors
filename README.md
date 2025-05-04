# Bayesian Changepoint Detection in Investor Behavior

We aim to detect structural changes in investor behavior toward the QQQ ETF by modeling daily transaction count data as a time series of Poisson-distributed observations. These changepoints may reflect shifts in market sentiment, news-driven reactions, or behavioral regime switches among retail or institutional investors.

## Methodology

We employ a Bayesian changepoint model using a Gibbs sampler to infer:

- The number and location of changepoints
- Poisson rates within each segment
- Posterior distributions for model parameters

The model assumes that the observed counts \( y_1, y_2, ..., y_T \) are generated from a piecewise Poisson process, with segment-specific rates \( \lambda_1, \lambda_2, ..., \lambda_K \) for K changepoints. A conjugate Gamma prior is used for each Poisson rate, and changepoint locations are assigned a uniform prior.

## Implementation

The model is implemented in R using custom Gibbs sampling steps:

- Segment boundaries are updated using conditional posterior distributions.
- Poisson rates are sampled from their full conditional Gamma distributions.
- Posterior summaries (mean changepoint locations, credible intervals) are derived from the Gibbs output.

## Preliminary Results

The model detects significant changepoints in the QQQ transaction time series, particularly in periods coinciding with major macroeconomic

## Next Steps

- Refine priors using empirical Bayes or hierarchical modeling.
- Compare models with varying numbers of changepoints using marginal likelihood or WAIC.
- Extend to non-homogeneous Poisson or Negative Binomial models to account for overdispersion.
- Visualize changepoint posterior samples and overlay them with price/volume series.

