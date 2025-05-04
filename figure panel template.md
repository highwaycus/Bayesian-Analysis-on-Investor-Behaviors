# Figure Panel: Bayesian Changepoint Detection Results

## Figure 1: Daily QQQ Transaction Counts
![Figure 1: QQQ Transaction Counts](figures/figure1_transaction_counts.png)  
*Time series of daily transaction counts for the QQQ ETF. Observed counts show notable variability, with candidate structural shifts.*

---

## Figure 2: Posterior Distribution of Changepoints
![Figure 2: Posterior Changepoint Locations](figures/figure2_changepoints_posterior.png)  
*Posterior probabilities of changepoint locations. Peaks indicate high posterior density regions corresponding to behavioral regime shifts.*

---

## Figure 3: Estimated Poisson Rates per Segment
![Figure 3: Estimated Poisson Rates](figures/figure3_lambda_segments.png)  
*Posterior means and 95% credible intervals for Poisson rate parameters across each segment between changepoints.*

---

## Figure 4: Overlay of Changepoints on Price Series
![Figure 4: Changepoints on Price](figures/figure4_price_overlay.png)  
*QQQ closing price with inferred changepoint locations overlaid. Useful for comparing timing of structural breaks with market dynamics.*

---

## Figure 5: Model Fit vs Observed Data
![Figure 5: Model Fit](figures/figure5_model_fit.png)  
*Posterior predictive checks: observed transaction counts versus model-based expected counts with uncertainty bands.*

