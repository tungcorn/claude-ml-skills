---
name: statsmodels
description: Statistical models library for Python. Use when you need specific model classes (OLS, GLM, mixed models, ARIMA) with detailed diagnostics, residuals, and inference. Best for econometrics, time series, rigorous inference with coefficient tables. For guided statistical test selection with APA reporting use statistical-analysis.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Statsmodels: Statistical Modeling and Econometrics

## Overview

Statsmodels is Python's premier library for statistical modeling, providing tools for estimation, inference, and diagnostics across a wide range of statistical methods. Apply this skill for rigorous statistical analysis, from simple linear regression to complex time series models and econometric analyses.

## When to Use This Skill

This skill should be used when:
- Fitting regression models (OLS, WLS, GLS, quantile regression)
- Performing generalized linear modeling (logistic, Poisson, Gamma, etc.)
- Analyzing discrete outcomes (binary, multinomial, count, ordinal)
- Conducting time series analysis (ARIMA, SARIMAX, VAR, forecasting)
- Running statistical tests and diagnostics
- Testing model assumptions (heteroskedasticity, autocorrelation, normality)
- Detecting outliers and influential observations
- Comparing models (AIC/BIC, likelihood ratio tests)
- Estimating causal effects
- Producing publication-ready statistical tables and inference

## Quick Start Guide

### Linear Regression (OLS)

```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Prepare data - ALWAYS add constant for intercept
X = sm.add_constant(X_data)

# Fit OLS model
model = sm.OLS(y, X)
results = model.fit()

# View comprehensive results
print(results.summary())

# Key results
print(f"R-squared: {results.rsquared:.4f}")
print(f"Coefficients:\\n{results.params}")
print(f"P-values:\\n{results.pvalues}")

# Predictions with confidence intervals
predictions = results.get_prediction(X_new)
pred_summary = predictions.summary_frame()
print(pred_summary)  # includes mean, CI, prediction intervals

# Diagnostics
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(results.resid, X)
print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")

# Visualize residuals
import matplotlib.pyplot as plt
plt.scatter(results.fittedvalues, results.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()
```

### Logistic Regression (Binary Outcomes)

```python
from statsmodels.discrete.discrete_model import Logit

# Add constant
X = sm.add_constant(X_data)

# Fit logit model
model = Logit(y_binary, X)
results = model.fit()

print(results.summary())

# Odds ratios
odds_ratios = np.exp(results.params)
print("Odds ratios:\\n", odds_ratios)

# Predicted probabilities
probs = results.predict(X)

# Binary predictions (0.5 threshold)
predictions = (probs > 0.5).astype(int)

# Model evaluation
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_binary, predictions))
print(f"AUC: {roc_auc_score(y_binary, probs):.4f}")

# Marginal effects
marginal = results.get_margeff()
print(marginal.summary())
```

### Time Series (ARIMA)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Check stationarity
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(y_series)
print(f"ADF p-value: {adf_result[1]:.4f}")

if adf_result[1] > 0.05:
    # Series is non-stationary, difference it
    y_diff = y_series.diff().dropna()

# Plot ACF/PACF to identify p, q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y_diff, lags=40, ax=ax1)
plot_pacf(y_diff, lags=40, ax=ax2)
plt.show()

# Fit ARIMA(p,d,q)
model = ARIMA(y_series, order=(1, 1, 1))
results = model.fit()

print(results.summary())

# Forecast
forecast = results.forecast(steps=10)
forecast_obj = results.get_forecast(steps=10)
forecast_df = forecast_obj.summary_frame()

print(forecast_df)  # includes mean and confidence intervals

# Residual diagnostics
results.plot_diagnostics(figsize=(12, 8))
plt.show()
```

### Generalized Linear Models (GLM)

```python
import statsmodels.api as sm

# Poisson regression for count data
X = sm.add_constant(X_data)
model = sm.GLM(y_counts, X, family=sm.families.Poisson())
results = model.fit()

print(results.summary())

# Rate ratios (for Poisson with log link)
rate_ratios = np.exp(results.params)
print("Rate ratios:\\n", rate_ratios)

# Check overdispersion
overdispersion = results.pearson_chi2 / results.df_resid
print(f"Overdispersion: {overdispersion:.2f}")

if overdispersion > 1.5:
    # Use Negative Binomial instead
    from statsmodels.discrete.count_model import NegativeBinomial
    nb_model = NegativeBinomial(y_counts, X)
    nb_results = nb_model.fit()
    print(nb_results.summary())
```

## Core Statistical Modeling Capabilities

### 1. Linear Regression Models

Comprehensive suite of linear models for continuous outcomes with various error structures.

**Available models:**
- **OLS**: Standard linear regression with i.i.d. errors
- **WLS**: Weighted least squares for heteroskedastic errors
- **GLS**: Generalized least squares for arbitrary covariance structure
- **GLSAR**: GLS with autoregressive errors for time series
- **Quantile Regression**: Conditional quantiles (robust to outliers)
- **Mixed Effects**: Hierarchical/multilevel models with random effects
- **Recursive/Rolling**: Time-varying parameter estimation

**Key features:**
- Comprehensive diagnostic tests
- Robust standard errors (HC, HAC, cluster-robust)
- Influence statistics (Cook's distance, leverage, DFFITS)
- Hypothesis testing (F-tests, Wald tests)
- Model comparison (AIC, BIC, likelihood ratio tests)
- Prediction with confidence and prediction intervals

**When to use:** Continuous outcome variable, want inference on coefficients, need diagnostics

**Reference:** See `references/linear_models.md` for detailed guidance on model selection, diagnostics, and best practices.

### 2. Generalized Linear Models (GLM)

Flexible framework extending linear models to non-normal distributions.

**Distribution families:**
- **Binomial**: Binary outcomes or proportions (logistic regression)
- **Poisson**: Count data
- **Negative Binomial**: Overdispersed counts
- **Gamma**: Positive continuous, right-skewed data
- **Inverse Gaussian**: Positive continuous with specific variance structure
- **Gaussian**: Equivalent to OLS
- **Tweedie**: Flexible family for semi-continuous data

**Link functions:**
- Logit, Probit, Log, Identity, Inverse, Sqrt, CLogLog, Power
- Choose based on interpretation needs and model fit

**Key features:**
- Maximum likelihood estimation via IRLS
- Deviance and Pearson residuals
- Goodness-of-fit statistics
- Pseudo R-squared measures
- Robust standard errors

**When to use:** Non-normal outcomes, need flexible variance and link specifications

**Reference:** See `references/glm.md` for family selection, link functions, interpretation, and diagnostics.

### 3. Discrete Choice Models

Models for categorical and count outcomes.

**Binary models:**
- **Logit**: Logistic regression (odds ratios)
- **Probit**: Probit regression (normal distribution)

**Multinomial models:**
- **MNLogit**: Unordered categories (3+ levels)
- **Conditional Logit**: Choice models with alternative-specific variables
- **Ordered Model**: Ordinal outcomes (ordered categories)

**Count models:**
- **Poisson**: Standard count model
- **Negative Binomial**: Overdispersed counts
- **Zero-Inflated**: Excess zeros (ZIP, ZINB)
- **Hurdle Models**: Two-stage models for zero-heavy data

**Key features:**
- Maximum likelihood estimation
- Marginal effects at means or average marginal effects
- Model comparison via AIC/BIC
- Predicted probabilities and classification
- Goodness-of-fit tests

**When to use:** Binary, categorical, or count outcomes

**Reference:** See `references/discrete_choice.md` for model selection, interpretation, and evaluation.

### 4. Time Series Analysis

Comprehensive time series modeling and forecasting capabilities.

**Univariate models:**
- **AutoReg (AR)**: Autoregressive models
- **ARIMA**: Autoregressive integrated moving average
- **SARIMAX**: Seasonal ARIMA with exogenous variables
- **Exponential Smoothing**: Simple, Holt, Holt-Winters
- **ETS**: Innovations state space models

**Multivariate models:**
- **VAR**: Vector autoregression
- **VARMAX**: VAR with MA and exogenous variables
- **Dynamic Factor Models**: Extract common factors
- **VECM**: Vector error correction models (cointegration)

**Advanced models:**
- **State Space**: Kalman filtering, custom specifications
- **Regime Switching**: Markov switching models
- **ARDL**: Autoregressive distributed lag

**Key features:**
- ACF/PACF analysis for model identification
- Stationarity tests (ADF, KPSS)
- Forecasting with prediction intervals
- Residual diagnostics (Ljung-Box, heteroskedasticity)
- Granger causality testing
- Impulse response functions (IRF)
- Forecast error variance decomposition (FEVD)

**When to use:** Time-ordered data, forecasting, understanding temporal dynamics

**Reference:** See `references/time_series.md` for model selection, diagnostics, and forecasting methods.

### 5. Statistical Tests and Diagnostics

Extensive testing and diagnostic capabilities for model validation.

**Residual diagnostics:**
- Autocorrelation tests (Ljung-Box, Durbin-Watson, Breusch-Godfrey)
- Heteroskedasticity tests (Breusch-Pagan, White, ARCH)
- Normality tests (Jarque-Bera, Omnibus, Anderson-Darling, Lilliefors)
- Specification tests (RESET, Harvey-Collier)

**Influence and outliers:**
- Leverage (hat values)
- Cook's distance
- DFFITS and DFBETAs
- Studentized residuals
- Influence plots

**Hypothesis testing:**
- t-tests (one-sample, two-sample, paired)
- Proportion tests
- Chi-square tests
- Non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
- ANOVA (one-way, two-way, repeated measures)

**Multiple comparisons:**
- Tukey's HSD
- Bonferroni correction
- False Discovery Rate (FDR)

**Effect sizes and power:**
- Cohen's d, eta-squared
- Power analysis for t-tests, proportions
- Sample size calculations

**Robust inference:**
- Heteroskedasticity-consistent SEs (HC0-HC3)
- HAC standard errors (Newey-West)
- Cluster-robust standard errors

**When to use:** Validating assumptions, detecting problems, ensuring robust inference

**Reference:** See `references/stats_diagnostics.md` for comprehensive testing and diagnostic procedures.

## Formula API (R-style)

Statsmodels supports R-style formulas for intuitive model specification:

```python
import statsmodels.formula.api as smf

# OLS with formula
results = smf.ols('y ~ x1 + x2 + x1:x2', data=df).fit()

# Categorical variables (automatic dummy coding)
results = smf.ols('y ~ x1 + C(category)', data=df).fit()

# Interactions
results = smf.ols('y ~ x1 * x2', data=df).fit()  # x1 + x2 + x1:x2

# Polynomial terms
results = smf.ols('y ~ x + I(x**2)', data=df).fit()

# Logit
results = smf.logit('y ~ x1 + x2 + C(group)', data=df).fit()

# Poisson
results = smf.poisson('count ~ x1 + x2', data=df).fit()

# ARIMA (not available via formula, use regular API)
```

## Model Selection and Comparison

### Information Criteria

```python
# Compare models using AIC/BIC
models = {
    'Model 1': model1_results,
    'Model 2': model2_results,
    'Model 3': model3_results
}

comparison = pd.DataFrame({
    'AIC': {name: res.aic for name, res in models.items()},
    'BIC': {name: res.bic for name, res in models.items()},
    'Log-Likelihood': {name: res.llf for name, res in models.items()}
})

print(comparison.sort_values('AIC'))
# Lower AIC/BIC indicates better model
```

### Likelihood Ratio Test (Nested Models)

```python
# For nested models (one is subset of the other)
from scipy import stats

lr_stat = 2 * (full_model.llf - reduced_model.llf)
df = full_model.df_model - reduced_model.df_model
p_value = 1 - stats.chi2.cdf(lr_stat, df)

print(f"LR statistic: {lr_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Full model significantly better")
else:
    print("Reduced model preferred (parsimony)")
```

### Cross-Validation

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Fit model
    model = sm.OLS(y_train, X_train).fit()

    # Predict
    y_pred = model.predict(X_val)

    # Score
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    cv_scores.append(rmse)

print(f"CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

## Best Practices

- **Data prep** — always `sm.add_constant()` for intercept; handle missing values; scale if needed for convergence; encode categoricals via formula API or dummy coding.
- **Model building** — start simple and add complexity only as needed; check assumptions (residuals, heteroskedasticity, autocorrelation); match model to outcome type (binary → `Logit`, count → `Poisson`); switch to robust methods or alternative model if assumptions break.
- **Inference** — report effect sizes alongside p-values; use robust SEs (HC/HAC/cluster) when heteroskedasticity or clustering present; correct for multiple comparisons; always include confidence intervals.
- **Evaluation** — plot residuals vs fitted + Q-Q; check Cook's distance / leverage / DFFITS; validate on holdout or via CV; compare via AIC/BIC (non-nested) or LR test (nested).
- **Reporting** — use `.summary()` for full output; document transformations and excluded observations; interpret per link function (`exp(β)` for log link); visualize predictions + CI + diagnostics.

## Common Workflows

- **Linear Regression Analysis** — EDA → fit OLS → residual diagnostics → heteroskedasticity / autocorrelation tests → VIF for multicollinearity → influence diagnostics → robust SEs if needed → interpret → validate (holdout or CV).
- **Binary Classification** — fit `Logit` → check convergence → interpret odds ratios → marginal effects → AUC + confusion matrix → influence check → compare with `Probit` → holdout validation.
- **Count Data Analysis** — fit Poisson → check overdispersion → switch to Negative Binomial if needed → check zero-inflation (ZIP/ZINB) → interpret rate ratios → GoF → AIC compare → validate.
- **Time Series Forecasting** — plot for trend/seasonality → ADF/KPSS stationarity → difference if needed → ACF/PACF for `p, q` → fit ARIMA/SARIMAX → Ljung-Box on residuals → forecast with CI → evaluate on test set.

## Reference Documentation

- **`references/linear_models.md`** — OLS, WLS, GLS, GLSAR, quantile regression, mixed effects, recursive/rolling. Covers diagnostics (heteroskedasticity, autocorrelation, multicollinearity, influence), robust SEs (HC/HAC/cluster), hypothesis testing, model comparison.
- **`references/glm.md`** — every distribution family (Binomial, Poisson, NegBin, Gamma, IG, Tweedie), link functions, IRLS fitting, deviance/Pearson residuals, pseudo R².
- **`references/discrete_choice.md`** — binary (Logit/Probit), multinomial (MNLogit, Conditional), ordinal, count (Poisson/NegBin/ZIP/ZINB/Hurdle); marginal effects and interpretation.
- **`references/time_series.md`** — univariate (AR, ARIMA, SARIMAX, ETS), multivariate (VAR, VARMAX, VECM, Dynamic Factor), state space, stationarity tests, forecast evaluation, Granger/IRF/FEVD.
- **`references/stats_diagnostics.md`** — residual diagnostics (autocorr, heteroskedasticity, normality), influence/outliers, parametric and non-parametric tests, ANOVA, multiple-comparison correction, robust covariance, power analysis.

Load a reference for parameter detail, similar-model comparison, troubleshooting, or advanced features. Use `grep -r "<term>" references/` to locate specific tests or models.

## Common Pitfalls to Avoid

1. **Forgetting constant term**: Always use `sm.add_constant()` unless no intercept desired
2. **Ignoring assumptions**: Check residuals, heteroskedasticity, autocorrelation
3. **Wrong model for outcome type**: Binary→Logit/Probit, Count→Poisson/NB, not OLS
4. **Not checking convergence**: Look for optimization warnings
5. **Misinterpreting coefficients**: Remember link functions (log, logit, etc.)
6. **Using Poisson with overdispersion**: Check dispersion, use Negative Binomial if needed
7. **Not using robust SEs**: When heteroskedasticity or clustering present
8. **Overfitting**: Too many parameters relative to sample size
9. **Data leakage**: Fitting on test data or using future information
10. **Not validating predictions**: Always check out-of-sample performance
11. **Comparing non-nested models**: Use AIC/BIC, not LR test
12. **Ignoring influential observations**: Check Cook's distance and leverage
13. **Multiple testing**: Correct p-values when testing many hypotheses
14. **Not differencing time series**: Fit ARIMA on non-stationary data
15. **Confusing prediction vs confidence intervals**: Prediction intervals are wider

## Getting Help

For detailed documentation and examples:
- Official docs: https://www.statsmodels.org/stable/
- User guide: https://www.statsmodels.org/stable/user-guide.html
- Examples: https://www.statsmodels.org/stable/examples/index.html
- API reference: https://www.statsmodels.org/stable/api.html

