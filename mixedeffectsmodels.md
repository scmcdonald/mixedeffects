Checking Mixed Effects Model
================
2024-07-01

# Model

Mixed effects model with two fixed effects (including an intercept) and
correlation within individual.

$$y_{it} = \beta_0 + x_{it}\beta_1  +  \gamma_i + \epsilon_{it}$$ This
is the matrix form with $3$ individuals and $2$ obserations per
individual.

$$
\begin{bmatrix}
y_{11} \\
y_{12} \\
y_{21} \\
y_{22} \\
y_{31} \\
y_{32}
\end{bmatrix} = 
\beta_0 \begin{bmatrix}
1 \\
1 \\
1 \\
1 \\
1 \\
1
\end{bmatrix} +\beta_1 \begin{bmatrix}
x_{11} \\
x_{12} \\
x_{21} \\
x_{22} \\
x_{31} \\
x_{32}
\end{bmatrix} + 
\begin{bmatrix}
1 & 0 & 0 \\
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\gamma_1 \\
\gamma_2 \\
\gamma_3
\end{bmatrix} +
\begin{bmatrix}
\epsilon_{11} \\
\epsilon_{12} \\
\epsilon_{21} \\
\epsilon_{22} \\
\epsilon_{31} \\
\epsilon_{32}
\end{bmatrix}
$$

with $\gamma_i \sim N(0, \sigma^2_\gamma)$ and
$\epsilon_i \sim N(0, \sigma^2_\epsilon)$.

## Data Simulation Functions

First we need a function that produces the intra-individual correlation
design matrix $\mathbf{Z}$:

``` r
# get intra-individual correlation design matrix
# kronecker product produces this

generate_random_effects_block_matrix <- function(n, m) {
  
  # create a block of ones of size m x 1
  block <- matrix(1, nrow = m, ncol = 1)
  
  # get z matrix
  z <- kronecker(diag(n), block)
  
  return(z)
}

# test
generate_random_effects_block_matrix(3, 2)
```

    ##      [,1] [,2] [,3]
    ## [1,]    1    0    0
    ## [2,]    1    0    0
    ## [3,]    0    1    0
    ## [4,]    0    1    0
    ## [5,]    0    0    1
    ## [6,]    0    0    1

We will generate data as follows:

$$\mathbf{x} = \begin{bmatrix} x_{11} \\
x_{12} \\
... \\
x_{1m} \\
... \\
x_{n1} \\
... \\
x_{nm}\\
\end{bmatrix} \sim MVN(0, I_{nm})$$

$\pmb{\gamma} \sim MVN(0, \sigma_u^2 I_n)$

$\pmb{\epsilon} \sim MVN(0, \sigma_e^2 I_{nm})$

``` r
generate_data <- function(n, m, beta0, beta1, sigma_u, sigma_e) {
  #individual
  id <- rep(1:n, each = m)
  
  # design matrix
  x <- rmvnorm(m*n, mean = 0, diag(1))
  
  # random effects design matrix
  z <- generate_random_effects_block_matrix(n, m)
  
  # random effect
  u <- rmvnorm(n, 0, sigma_u * diag(1))
  
  # error
  epsilon <- rmvnorm(n*m, 0, sigma_e * diag(1))
  
  # calculate y
  y <- beta0 + beta1 * x + z %*% u + epsilon
  
  # return data
  data.frame(y = y, x = x, id = id)
}

# test
generate_data(n = 3, m = 2, beta0 = 1, beta1 = 2, sigma_u = 1, sigma_e = 1)
```

    ##            y          x id
    ## 1  0.6989395  0.5499558  1
    ## 2 -1.7039162 -0.8615706  1
    ## 3  2.1029347  0.9149381  2
    ## 4 -0.6054760  0.2678015  2
    ## 5 -1.2023469 -0.8633746  3
    ## 6 -1.3750198 -1.6511244  3

# New work - Steps

1.  Fit model using `lme4`, using the *REML = F* to optimize
    log-likelihood.

2.  Extract
    $\hat\theta_{MLE} = (\hat\beta_0, \hat\beta_1, \hat\sigma_{\gamma}^2, \hat\sigma_{\epsilon}^2)$
    from the model output.

3.  Plug into function thtat computes
    $\hat{\Sigma} = \frac{1}{n} J(\hat\theta_{MLE})^{-1}$ by taking the
    Hessian of the log-likelihood, plugs in our MLE, and divided by $n$.

Note that first I had to update the log-likelihood as a function of n
individual and m observations per user, because I had previously just
done it for $n = 3$ amd $m =2$. The final answer is below and the work
is in the [overleaf
file](https://www.overleaf.com/read/jmnzvhwsnwqq#81156c) section 5.1.1.

$$\ell = -\frac{nm}{2} \log(2\pi) - \frac{n}{2} \left((m-1)\log(\sigma^2_\epsilon) + \log( \sigma^2_\epsilon + m \sigma_\gamma^2) \right)$$

$$-\frac{1}{2\sigma_\epsilon^2} \left[ \sum_{i = 1}^n \sum_{j = 1}^m (y_{ij}  - \beta_0 - \beta_1 x_{ij})^2 -  \frac{\sigma_\gamma^{2} }{\sigma_\epsilon^{2} + m\sigma^{2}_\gamma }  \sum_{i = 1}^n \left(\sum_{j= 1}^m (y_{ij} - \beta_0 - \beta_1 x_{ij})\right)^2 \right]$$

## Simulation

``` r
set.seed(1)

# generate data
data <- generate_data(n = 50, m = 20, 
                      beta0 = 1, beta1 = 2, 
                      sigma_u = 1, sigma_e = 1)

# fit model
mod <- lmer(y ~ x + (1 | id), data = data, REML = FALSE)
summary(mod)
```

    ## Linear mixed model fit by maximum likelihood  ['lmerMod']
    ## Formula: y ~ x + (1 | id)
    ##    Data: data
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##   3073.8   3093.4  -1532.9   3065.8      996 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.0999 -0.6552 -0.0299  0.6969  3.1910 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev.
    ##  id       (Intercept) 1.020    1.01    
    ##  Residual             1.082    1.04    
    ## Number of obs: 1000, groups:  id, 50
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error t value
    ## (Intercept)   0.9586     0.1466    6.54
    ## x             1.9746     0.0324   60.95
    ## 
    ## Correlation of Fixed Effects:
    ##   (Intr)
    ## x 0.003

``` r
# extract MLEs

# fixed effects
fixef(mod)
```

    ## (Intercept)           x 
    ##   0.9585606   1.9746153

``` r
beta_hats <- unname(fixef(mod))

# variance of the random effect 
VarCorr(mod)
```

    ##  Groups   Name        Std.Dev.
    ##  id       (Intercept) 1.01    
    ##  Residual             1.04

``` r
VarCorr(mod)$id[1]
```

    ## [1] 1.02002

``` r
sigma_u2_hat <- VarCorr(mod)$id[1]^2

# variance of the error
sigma_e2_hat <- summary(mod)$sigma^2

sigma_e2_hat
```

    ## [1] 1.081648

``` r
# define log likelihood function for differentiation

ll <- function(beta0, beta1, su2, se2,n = 50,m = 20) {

  term1 <- -(n/2) * ((m-1)*log(se2) + log(se2 + m *su2))
  
  data$eq <- data$y - beta0 - (beta1 * data$x)

  sum_within_id <- aggregate(eq ~ id, data = data, FUN = sum)
  
  term2 <- -(1/(2 * se2)) * ( sum(data$eq^2) 
    - ( (su2 / (se2 + m * su2)) * sum(sum_within_id$eq^2))
    )
  
  return(term1 + term2)
  
}

obs_info <- - calculus::hessian(f = ll, var = c("beta0" =beta_hats[1] , 
                                                "beta1" =beta_hats[2], 
                                                "su2" = sigma_u2_hat, 
                                                "se2" = sigma_e2_hat))


est_sigma <- solve(obs_info)/50


dimnames(est_sigma) <- list(c("beta0", "beta1", "su2", "se2"), c("beta0", "beta1", "su2", "se2"))

est_sigma
```

    ##               beta0         beta1           su2           se2
    ## beta0  4.378124e-04  2.445372e-07  1.853727e-09 -9.433538e-11
    ## beta1  2.445372e-07  2.099366e-05  1.591260e-07 -8.098507e-09
    ## su2    1.853727e-09  1.591260e-07  9.956595e-04 -2.463139e-06
    ## se2   -9.433538e-11 -8.098507e-09 -2.463139e-06  4.926156e-05

## Questions

- I am concerned that some of the output are negative.

- When using the estimate $J(\theta)^{-1}/n$, not sure if the $n$ here
  should be number of individuals, number of observations per
  individual, or toatl number of observations(number of individuals
  (times) number of observations per individual). Using number of
  individuals for now.

- I use a numeric differentiation package called `calculus` to calculate
  the Hessian. I read that automatic differentiation is usually better,
  but there doesn’t seem to be great packages to do this in R. It seems
  like Python might have good packages for this.

<!-- You actually should get a similar answer by conducting 1000 simulated trials each of n users, estimating the fixed effects and the random effects variance for each trial (so you get 1000 of these)  and then calculating sample variances, sample covariances for the 1000 vectors (each vector contains the estimators of the fixed effects, estimators of the random effects variances and estimators of the noise variance). -->
<!-- ```{r} -->
<!-- out <- replicate(1000, { -->
<!--   data <- generate_data(n = 50, m = 20,  -->
<!--                       beta0 = 1, beta1 = 2,  -->
<!--                       sigma_u = 1, sigma_e = 1) -->
<!--         # REML = F ensures we run ML not REML -->
<!--       model <- lmer(y ~ x + (1 | id), data = data, REML = FALSE) -->
<!--       # extract estimates -->
<!--       return(c(beta_0 = fixef(model)[1],  -->
<!--                beta1 = fixef(model)[2],  -->
<!--                sigma_u = VarCorr(model)$id[1])) -->
<!-- } -->
<!-- ) -->
<!-- cov(t(out)) -->
<!-- ``` -->
<!-- # Old -->
<!-- ## Simulations -->
<!-- Note that I am running a mixed effects model with random intercept, but not random slope.  -->
<!-- ```{r} -->
<!-- # simulation -->
<!-- simulation <- function(ns, m, beta0, beta1, sigma_u, sigma_e, num_simulations){ -->
<!--   # we want to construct a dataframe with number of individuals and -->
<!--   # correlation for beta0 vs sigma_u, correlation for beta1 vs sigma_u -->
<!--   out <- data.frame(n = rep(ns, each = 2),  -->
<!--                     beta_vs_sigmau = rep(c("beta0_vs_sigmau", "beta1_vs_sigmau"),  length(ns)),  -->
<!--                     corr = NA_real_) -->
<!--   # repeat for each individual size n -->
<!--   for(i in ns){ -->
<!--     # within each individual, repeat (1000?) times -->
<!--     sims <- replicate(num_simulations, { -->
<!--       # simulate data -->
<!--       data <- generate_data(i, m, beta0, beta1, sigma_u, sigma_e) -->
<!--       # fit model - this is random intercept, but not random slope -->
<!--       # REML = F ensures we run ML not REML -->
<!--       model <- lmer(y ~ x + (1 | id), data = data, REML = FALSE) -->
<!--       # extract estimates -->
<!--       return(c(beta_0 = fixef(model)[1],  -->
<!--                beta1 = fixef(model)[2],  -->
<!--                sigma_u = VarCorr(model)$id[1])) -->
<!--     }) -->
<!--     # get correlations -->
<!--     corrs <- cor(t(sims)) -->
<!--     # 2 is beta0 vs sigmau, 3 is beta1 vs sigmau -->
<!--     corrs <- corrs[lower.tri(corrs)][2:3] -->
<!--     # store values -->
<!--     out[out$n == i, 3] <- corrs -->
<!--   } -->
<!--   gg <- ggplot(out, aes(x = n, group = beta_vs_sigmau, y = corr, color = beta_vs_sigmau)) +  -->
<!--     geom_hline(yintercept = 0, color = "grey70", linetype = "dashed") + -->
<!--     geom_line() +  -->
<!--     geom_point() +  -->
<!--     theme_minimal() +  -->
<!--     labs(y = "Correlation", x = "Number of Individuals", color = "Comparison") -->
<!--   print(gg) -->
<!-- } -->
<!-- ``` -->
<!-- For each number of individual $n = \{10,  25, 50, 75, 100, 150, 200, 250,  300, 350, 400, 450, 500\}$, we will generate data and fit our model 1000 times. Then we will take the correlation of those simulations.  -->
<!-- For each simulation, we fix that each individual has $m = 20$ observations and we will fix $\sigma_\epsilon^2 = 1$, $\beta_0 = 1$. -->
<!-- We will vary $\beta_1$ and $\sigma_\gamma^2$. -->
<!-- ### Case 1: Equal Beta1 and sigma gamma -->
<!-- Here we set $\beta_1 = \sigma^2_\gamma = 1$. -->
<!-- ```{r} -->
<!-- ns <- c(10,  25, 50, 75, 100, 150, 200, 250,  300, 350, 400, 450, 500) -->
<!-- nsims = 1000 -->
<!-- m = 20 -->
<!-- beta0 = 1 -->
<!-- sigma_e = 1 -->
<!-- set.seed(1) -->
<!-- ``` -->
<!-- ```{r,eval = F} -->
<!-- sim0 <- simulation(n = ns,  -->
<!--            m = m, beta0 = beta0, sigma_e = sigma_e, -->
<!--            beta1 = 1, sigma_u = 1,  -->
<!--            num_simulations =nsims) -->
<!-- saveRDS(sim0, file = "output/sim0.rds") -->
<!-- ``` -->
<!-- ```{r} -->
<!-- readRDS(file ="output/sim0.rds" ) -->
<!-- ``` -->
<!-- ### Large Beta1  -->
<!-- Here we set $\beta_1 = 10$ and $\sigma^2_\gamma = 1$. -->
<!-- ```{r, eval = F} -->
<!-- sim1 <- simulation(n =ns, m = m, beta0 = beta0, beta1 = 10,  -->
<!--            sigma_u = 1, sigma_e = sigma_e, num_simulations =nsims) -->
<!-- saveRDS(sim1, file = "output/sim1.rds") -->
<!-- ``` -->
<!-- ```{r} -->
<!-- readRDS(file ="output/sim1.rds" ) -->
<!-- ``` -->
<!-- ### Large sigma_u  -->
<!-- Here we set $\beta_1 = 1$ and $\sigma^2_\gamma = 10$. -->
<!-- ```{r, eval = F} -->
<!-- sim2 <- simulation(n = ns, m = m, beta0 = beta0, beta1 = 1,  -->
<!--            sigma_u = 10, sigma_e = sigma_e, num_simulations =nsims) -->
<!-- saveRDS(sim2, file = "output/sim2.rds") -->
<!-- ``` -->
<!-- ```{r} -->
<!-- readRDS(file ="output/sim2.rds" ) -->
<!-- ``` -->
<!-- ### Large sigma_u and beta1 -->
<!-- Here we set $\beta_1 = 10$ and $\sigma^2_\gamma = 10$. -->
<!-- ```{r, eval = F} -->
<!-- sim3 <- simulation(n = ns, m = m, beta0 = beta0, beta1 = 10, -->
<!--            sigma_u = 10, sigma_e = sigma_e, num_simulations =nsims) -->
<!-- saveRDS(sim3, file = "output/sim3.rds") -->
<!-- ``` -->
<!-- ```{r} -->
<!-- readRDS(file ="output/sim3.rds" ) -->
<!-- ``` -->
<!-- ## Questions -->
<!-- - Sometimes I get errors regarding singular fit or failed to converge. I am not sure what this means or how to avoid it. -->
<!-- - I think I might try the mixed effects model with random intercept and random slope, to see if it makes a difference -->
<!-- - The take away is that $\sigma_\gamma^2$ and $\beta_1, \beta_0$ are correlated, but I am wondering if we should see a specific pattern as $n$ grows. For example, should we expect that as $n$ increase, correlation is stronger (and in one direction? -->
