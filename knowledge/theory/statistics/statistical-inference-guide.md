# Statistical Inference: Complete Theory and Practice Guide

## Table of Contents

1. [Foundations of Probability](#foundations-of-probability)
2. [Distributions and Their Properties](#distributions-and-their-properties)
3. [Sampling and Sampling Distributions](#sampling-and-sampling-distributions)
4. [Point Estimation](#point-estimation)
5. [Confidence Intervals](#confidence-intervals)
6. [Hypothesis Testing](#hypothesis-testing)
7. [Statistical Power and Sample Size](#statistical-power-and-sample-size)
8. [Nonparametric Methods](#nonparametric-methods)
9. [Bayesian Inference](#bayesian-inference)
10. [Multiple Comparisons and False Discovery Rate](#multiple-comparisons-and-false-discovery-rate)
11. [Advanced Topics](#advanced-topics)
12. [Practical Guidelines](#practical-guidelines)

## Foundations of Probability

### Basic Probability Concepts

#### Sample Space and Events
- **Sample Space (Ω)**: Set of all possible outcomes of an experiment
- **Event (A)**: Subset of the sample space
- **Elementary Event**: Single outcome
- **Compound Event**: Union or intersection of elementary events

#### Probability Axioms (Kolmogorov)
1. **Non-negativity**: P(A) ≥ 0 for any event A
2. **Normalization**: P(Ω) = 1
3. **Countable Additivity**: For mutually exclusive events A₁, A₂, ...:
   P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...

#### Key Probability Rules
- **Addition Rule**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- **Multiplication Rule**: P(A ∩ B) = P(A|B) × P(B) = P(B|A) × P(A)
- **Complement Rule**: P(A') = 1 - P(A)
- **Total Probability**: P(B) = Σᵢ P(B|Aᵢ) × P(Aᵢ)

#### Conditional Probability and Independence
- **Conditional Probability**: P(A|B) = P(A ∩ B) / P(B), provided P(B) > 0
- **Independence**: Events A and B are independent if P(A ∩ B) = P(A) × P(B)
- **Bayes' Theorem**: P(A|B) = [P(B|A) × P(A)] / P(B)

### Random Variables

#### Discrete Random Variables
- **Probability Mass Function (PMF)**: P(X = x) = p(x)
- **Properties**:
  - p(x) ≥ 0 for all x
  - Σₓ p(x) = 1
- **Cumulative Distribution Function**: F(x) = P(X ≤ x) = Σₜ≤ₓ p(t)

#### Continuous Random Variables
- **Probability Density Function (PDF)**: f(x)
- **Properties**:
  - f(x) ≥ 0 for all x
  - ∫₋∞^∞ f(x)dx = 1
  - P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx
- **Cumulative Distribution Function**: F(x) = P(X ≤ x) = ∫₋∞ˣ f(t)dt

#### Moments and Moment Generating Functions
- **Expected Value (Mean)**: E[X] = μ
  - Discrete: E[X] = Σₓ x × p(x)
  - Continuous: E[X] = ∫₋∞^∞ x × f(x)dx
- **Variance**: Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
- **Standard Deviation**: σ = √Var(X)
- **Moment Generating Function**: M(t) = E[e^(tX)]
- **Characteristic Function**: φ(t) = E[e^(itX)]

#### Properties of Expectation and Variance
- **Linearity of Expectation**: E[aX + bY + c] = aE[X] + bE[Y] + c
- **Variance Properties**:
  - Var(aX + c) = a²Var(X)
  - If X and Y are independent: Var(X + Y) = Var(X) + Var(Y)
- **Covariance**: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)] = E[XY] - E[X]E[Y]
- **Correlation**: ρ(X,Y) = Cov(X,Y) / (σₓσᵧ)

## Distributions and Their Properties

### Discrete Distributions

#### Bernoulli Distribution
- **Parameters**: p (probability of success)
- **PMF**: P(X = k) = p^k × (1-p)^(1-k), k ∈ {0,1}
- **Mean**: p
- **Variance**: p(1-p)
- **Applications**: Single trial experiments, binary outcomes

#### Binomial Distribution
- **Parameters**: n (number of trials), p (probability of success)
- **PMF**: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
- **Mean**: np
- **Variance**: np(1-p)
- **Applications**: Fixed number of independent trials
- **Approximations**: 
  - Normal when np > 5 and n(1-p) > 5
  - Poisson when n is large and p is small

#### Poisson Distribution
- **Parameter**: λ (rate parameter)
- **PMF**: P(X = k) = (λ^k × e^(-λ)) / k!
- **Mean**: λ
- **Variance**: λ
- **Applications**: Counting rare events, queuing theory
- **Relationship**: Limit of binomial as n → ∞, p → 0, np → λ

#### Geometric Distribution
- **Parameter**: p (probability of success)
- **PMF**: P(X = k) = (1-p)^(k-1) × p, k = 1,2,3,...
- **Mean**: 1/p
- **Variance**: (1-p)/p²
- **Applications**: Number of trials until first success
- **Memoryless Property**: P(X > s+t | X > s) = P(X > t)

#### Negative Binomial Distribution
- **Parameters**: r (number of successes), p (probability of success)
- **PMF**: P(X = k) = C(k+r-1, r-1) × p^r × (1-p)^k
- **Mean**: r(1-p)/p
- **Variance**: r(1-p)/p²
- **Applications**: Number of failures before r-th success

#### Hypergeometric Distribution
- **Parameters**: N (population size), K (successes in population), n (sample size)
- **PMF**: P(X = k) = [C(K,k) × C(N-K,n-k)] / C(N,n)
- **Mean**: n × K/N
- **Variance**: n × (K/N) × (1-K/N) × (N-n)/(N-1)
- **Applications**: Sampling without replacement

### Continuous Distributions

#### Normal (Gaussian) Distribution
- **Parameters**: μ (mean), σ² (variance)
- **PDF**: f(x) = (1/(σ√(2π))) × exp(-½((x-μ)/σ)²)
- **Properties**:
  - Symmetric around μ
  - Bell-shaped curve
  - 68-95-99.7 rule
- **Standard Normal**: Z ~ N(0,1)
- **Standardization**: Z = (X - μ)/σ
- **Applications**: Central Limit Theorem, many natural phenomena

#### Student's t-Distribution
- **Parameter**: ν (degrees of freedom)
- **PDF**: f(t) = [Γ((ν+1)/2) / (√(νπ)Γ(ν/2))] × (1 + t²/ν)^(-(ν+1)/2)
- **Properties**:
  - Symmetric around 0
  - Heavier tails than normal
  - Approaches normal as ν → ∞
- **Applications**: Small sample inference, confidence intervals

#### Chi-square Distribution
- **Parameter**: ν (degrees of freedom)
- **PDF**: f(x) = [1/(2^(ν/2)Γ(ν/2))] × x^(ν/2-1) × e^(-x/2), x > 0
- **Mean**: ν
- **Variance**: 2ν
- **Applications**: Goodness of fit tests, variance testing
- **Relationship**: Sum of ν independent standard normal squared variables

#### F-Distribution
- **Parameters**: ν₁, ν₂ (degrees of freedom)
- **PDF**: Complex form involving beta function
- **Mean**: ν₂/(ν₂-2) for ν₂ > 2
- **Applications**: ANOVA, comparing variances
- **Relationship**: Ratio of two independent chi-square variables

#### Exponential Distribution
- **Parameter**: λ (rate parameter)
- **PDF**: f(x) = λe^(-λx), x ≥ 0
- **CDF**: F(x) = 1 - e^(-λx)
- **Mean**: 1/λ
- **Variance**: 1/λ²
- **Applications**: Survival analysis, queuing theory
- **Memoryless Property**: P(X > s+t | X > s) = P(X > t)

#### Gamma Distribution
- **Parameters**: α (shape), β (rate)
- **PDF**: f(x) = (β^α/Γ(α)) × x^(α-1) × e^(-βx)
- **Mean**: α/β
- **Variance**: α/β²
- **Applications**: Waiting times, Bayesian priors
- **Special Cases**: Exponential (α=1), Chi-square (α=ν/2, β=1/2)

#### Beta Distribution
- **Parameters**: α, β (shape parameters)
- **PDF**: f(x) = [Γ(α+β)/(Γ(α)Γ(β))] × x^(α-1) × (1-x)^(β-1), 0 ≤ x ≤ 1
- **Mean**: α/(α+β)
- **Variance**: αβ/[(α+β)²(α+β+1)]
- **Applications**: Bayesian priors for probabilities, order statistics

#### Uniform Distribution
- **Parameters**: a, b (endpoints)
- **PDF**: f(x) = 1/(b-a), a ≤ x ≤ b
- **Mean**: (a+b)/2
- **Variance**: (b-a)²/12
- **Applications**: Random number generation, modeling ignorance

## Sampling and Sampling Distributions

### Types of Sampling

#### Probability Sampling
- **Simple Random Sampling**: Each unit has equal probability of selection
- **Stratified Sampling**: Population divided into strata, sampling within each
- **Cluster Sampling**: Population divided into clusters, sampling clusters
- **Systematic Sampling**: Every k-th unit selected after random start

#### Non-probability Sampling
- **Convenience Sampling**: Easily accessible units
- **Purposive Sampling**: Units selected based on judgment
- **Quota Sampling**: Fixed quotas for different groups
- **Snowball Sampling**: Referrals from selected units

### Sampling Distributions

#### Sample Mean Distribution
- **Population Normal**: X̄ ~ N(μ, σ²/n)
- **Population Non-normal**: By CLT, X̄ approximately N(μ, σ²/n) for large n
- **Standard Error**: SE(X̄) = σ/√n
- **Finite Population Correction**: SE(X̄) = (σ/√n) × √[(N-n)/(N-1)]

#### Sample Variance Distribution
- **For Normal Population**: (n-1)S²/σ² ~ χ²(n-1)
- **Expected Value**: E[S²] = σ²
- **Variance**: Var(S²) = 2σ⁴/(n-1)

#### Sample Proportion Distribution
- **Distribution**: p̂ approximately N(p, p(1-p)/n) for large n
- **Standard Error**: SE(p̂) = √[p(1-p)/n]
- **Rule of Thumb**: np ≥ 5 and n(1-p) ≥ 5

### Central Limit Theorem

#### Statement
For a sequence of independent, identically distributed random variables X₁, X₂, ..., Xₙ with mean μ and finite variance σ², the sample mean X̄ₙ converges in distribution to N(μ, σ²/n) as n → ∞.

#### Practical Implications
- **Sample Size**: Usually n ≥ 30 is sufficient for normal approximation
- **Skewed Populations**: May require larger sample sizes
- **Applications**: Basis for many statistical procedures

#### Berry-Esseen Theorem
Provides bound on convergence rate: |F(x) - Φ(x)| ≤ C × ρ/(σ³√n)
where ρ = E[|X - μ|³] and C ≈ 0.4748

### Law of Large Numbers

#### Weak Law (Convergence in Probability)
X̄ₙ →ᵖ μ as n → ∞

#### Strong Law (Almost Sure Convergence)
X̄ₙ →ᵃ·ˢ· μ as n → ∞

#### Practical Meaning
Sample averages converge to population mean with increasing sample size

## Point Estimation

### Properties of Estimators

#### Unbiasedness
- **Definition**: E[θ̂] = θ
- **Bias**: Bias(θ̂) = E[θ̂] - θ
- **Examples**: 
  - X̄ is unbiased for μ
  - S² is unbiased for σ²

#### Consistency
- **Weak Consistency**: θ̂ₙ →ᵖ θ as n → ∞
- **Strong Consistency**: θ̂ₙ →ᵃ·ˢ· θ as n → ∞
- **Mean Square Consistency**: E[(θ̂ₙ - θ)²] → 0 as n → ∞

#### Efficiency
- **Mean Squared Error**: MSE(θ̂) = Var(θ̂) + [Bias(θ̂)]²
- **Relative Efficiency**: e(θ̂₁, θ̂₂) = MSE(θ̂₂)/MSE(θ̂₁)
- **Asymptotic Efficiency**: Achieves Cramér-Rao lower bound asymptotically

#### Sufficiency
- **Definition**: T(X) is sufficient for θ if P(X|T,θ) doesn't depend on θ
- **Factorization Theorem**: T is sufficient iff f(x|θ) = g(T(x),θ) × h(x)
- **Minimal Sufficiency**: Sufficient statistic with smallest dimension

### Estimation Methods

#### Method of Moments
1. Set sample moments equal to population moments
2. Solve for parameters
3. **Example**: For normal distribution:
   - μ̂ = X̄
   - σ̂² = (1/n)Σ(Xᵢ - X̄)²

#### Maximum Likelihood Estimation (MLE)
1. Write likelihood function: L(θ) = ∏f(xᵢ|θ)
2. Take logarithm: ℓ(θ) = log L(θ)
3. Differentiate and set equal to zero: dℓ/dθ = 0
4. Solve for θ̂

**Properties of MLE**:
- Consistent under regularity conditions
- Asymptotically efficient
- Asymptotically normal
- Invariant under reparameterization

#### Bayesian Estimation
- **Prior Distribution**: π(θ)
- **Likelihood**: L(θ|x)
- **Posterior**: π(θ|x) ∝ L(θ|x) × π(θ)
- **Point Estimates**:
  - Posterior mean
  - Posterior median
  - Maximum a posteriori (MAP)

#### Least Squares Estimation
- **Objective**: Minimize Σ(yᵢ - f(xᵢ,θ))²
- **Normal Equations**: X'Xβ̂ = X'y
- **Solution**: β̂ = (X'X)⁻¹X'y (when X'X is invertible)

### Information and Efficiency

#### Fisher Information
- **Definition**: I(θ) = E[(-d²ℓ/dθ²)] = E[(dℓ/dθ)²]
- **Observed Information**: J(θ) = -d²ℓ/dθ²
- **Multivariate**: I(θ) is the Fisher information matrix

#### Cramér-Rao Lower Bound
- **Scalar Parameter**: Var(θ̂) ≥ 1/I(θ)
- **Vector Parameter**: Cov(θ̂) ≥ I(θ)⁻¹
- **Efficient Estimator**: Achieves the bound

#### Asymptotic Properties of MLE
- **Consistency**: θ̂ₙ →ᵖ θ₀
- **Asymptotic Normality**: √n(θ̂ₙ - θ₀) →ᵈ N(0, I(θ₀)⁻¹)
- **Asymptotic Efficiency**: Achieves Cramér-Rao bound

## Confidence Intervals

### Basic Concepts

#### Definition
A 100(1-α)% confidence interval for parameter θ is a random interval [L,U] such that P(L ≤ θ ≤ U) = 1-α before sampling.

#### Interpretation
- **Frequentist**: In repeated sampling, 100(1-α)% of intervals contain the true parameter
- **Not**: The probability that θ lies in a specific calculated interval is 1-α

#### Coverage Probability
Actual probability that interval contains true parameter value

### Confidence Intervals for Normal Populations

#### Mean with Known Variance
- **Interval**: X̄ ± z_(α/2) × σ/√n
- **Margin of Error**: E = z_(α/2) × σ/√n

#### Mean with Unknown Variance
- **Interval**: X̄ ± t_(α/2,n-1) × s/√n
- **Uses t-distribution with n-1 degrees of freedom

#### Variance
- **Interval**: [(n-1)s²/χ²_(α/2,n-1), (n-1)s²/χ²_(1-α/2,n-1)]
- **Uses chi-square distribution

#### Difference of Two Means
- **Equal Variances**: (X̄₁ - X̄₂) ± t_(α/2,n₁+n₂-2) × sₚ√(1/n₁ + 1/n₂)
- **Unequal Variances (Welch)**: (X̄₁ - X̄₂) ± t_(α/2,ν) × √(s₁²/n₁ + s₂²/n₂)
  where ν = (s₁²/n₁ + s₂²/n₂)²/[(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]

#### Ratio of Two Variances
- **Interval**: [s₁²/s₂² × 1/F_(α/2,n₁-1,n₂-1), s₁²/s₂² × F_(α/2,n₂-1,n₁-1)]

### Large Sample Confidence Intervals

#### Single Proportion
- **Wilson Score Interval**: More accurate for small samples
- **Wald Interval**: p̂ ± z_(α/2)√[p̂(1-p̂)/n]
- **Exact (Clopper-Pearson)**: Based on binomial distribution

#### Difference of Two Proportions
- **Interval**: (p̂₁ - p̂₂) ± z_(α/2)√[p̂₁(1-p̂₁)/n₁ + p̂₂(1-p̂₂)/n₂]

#### General Parameters (Delta Method)
If √n(θ̂ - θ) →ᵈ N(0,σ²), then for g(θ) with g'(θ) ≠ 0:
√n(g(θ̂) - g(θ)) →ᵈ N(0,[g'(θ)]²σ²)

### Advanced Confidence Interval Methods

#### Bootstrap Confidence Intervals
1. **Percentile Method**: Use α/2 and 1-α/2 quantiles of bootstrap distribution
2. **Bias-Corrected and Accelerated (BCa)**: Adjusts for bias and skewness
3. **Bootstrap-t**: Uses studentized bootstrap statistics

#### Likelihood-Based Intervals
- **Profile Likelihood**: Based on likelihood ratio test
- **Interval**: {θ: -2log[L(θ)/L(θ̂)] ≤ χ²_(α,1)}

#### Fiducial Intervals
- **Concept**: Invert test statistics to obtain intervals
- **Same as confidence intervals under certain conditions

## Hypothesis Testing

### Basic Concepts

#### Null and Alternative Hypotheses
- **Null Hypothesis (H₀)**: Statement of no effect or no difference
- **Alternative Hypothesis (H₁ or Hₐ)**: What we want to establish
- **Types**:
  - Two-sided: H₁: θ ≠ θ₀
  - One-sided: H₁: θ > θ₀ or H₁: θ < θ₀

#### Test Statistic
Function of sample data used to make decision about H₀

#### P-value
Probability of observing test statistic as extreme or more extreme than observed, assuming H₀ is true

#### Significance Level (α)
Probability of Type I error (rejecting true H₀)

#### Types of Errors
- **Type I Error**: Reject true H₀ (False Positive)
- **Type II Error**: Fail to reject false H₀ (False Negative)
- **Power**: 1 - P(Type II Error) = P(Reject H₀ | H₁ true)

### Classical Tests for Normal Populations

#### One-Sample t-test
- **Hypotheses**: H₀: μ = μ₀ vs H₁: μ ≠ μ₀
- **Test Statistic**: t = (X̄ - μ₀)/(s/√n)
- **Distribution**: t_(n-1) under H₀
- **Assumptions**: Normality, independence

#### Two-Sample t-tests
- **Equal Variances**: t = (X̄₁ - X̄₂)/(sₚ√(1/n₁ + 1/n₂))
- **Unequal Variances**: Welch's t-test
- **Paired t-test**: For dependent samples

#### F-test for Equal Variances
- **Test Statistic**: F = s₁²/s₂²
- **Distribution**: F_(n₁-1,n₂-1) under H₀

#### Chi-square Test for Variance
- **Test Statistic**: χ² = (n-1)s²/σ₀²
- **Distribution**: χ²_(n-1) under H₀

### Large Sample Tests

#### Z-test for Proportion
- **Test Statistic**: z = (p̂ - p₀)/√[p₀(1-p₀)/n]
- **Distribution**: Standard normal under H₀

#### Two-Proportion Z-test
- **Test Statistic**: z = (p̂₁ - p̂₂)/√[p̂(1-p̂)(1/n₁ + 1/n₂)]
- **Pooled Estimate**: p̂ = (x₁ + x₂)/(n₁ + n₂)

#### Wald Test
For large samples with θ̂ ~ N(θ, Σ):
- **Test Statistic**: W = (θ̂ - θ₀)'Σ⁻¹(θ̂ - θ₀)
- **Distribution**: χ²_p under H₀

### Likelihood Ratio Tests

#### Test Statistic
Λ = L(θ̂₀)/L(θ̂) where θ̂₀ is MLE under H₀, θ̂ is unrestricted MLE

#### Wilks' Theorem
-2log Λ →ᵈ χ²_r where r = dim(Θ) - dim(Θ₀)

#### Score Test (Lagrange Multiplier)
- **Test Statistic**: Based on score function at θ̂₀
- **Distribution**: χ²_r under H₀

### Goodness of Fit Tests

#### Pearson Chi-square Test
- **Test Statistic**: χ² = Σ(Oᵢ - Eᵢ)²/Eᵢ
- **Distribution**: Approximately χ²_(k-1-p) where p is number of estimated parameters

#### Kolmogorov-Smirnov Test
- **Test Statistic**: D = max|Fₙ(x) - F₀(x)|
- **Uses**: Testing distributional assumptions

#### Anderson-Darling Test
- **More Sensitive**: To differences in tails than KS test
- **Test Statistic**: Weighted squared difference between empirical and theoretical CDFs

#### Shapiro-Wilk Test
- **Specific for Normality**: More powerful than KS for normal distribution
- **Limitation**: Sample size restrictions

### Tests of Independence

#### Chi-square Test of Independence
- **Contingency Tables**: r × c tables
- **Test Statistic**: χ² = Σ(Oᵢⱼ - Eᵢⱼ)²/Eᵢⱼ
- **Distribution**: χ²_((r-1)(c-1)) under independence

#### Fisher's Exact Test
- **2×2 Tables**: When expected frequencies are small
- **Exact P-value**: Based on hypergeometric distribution

#### McNemar's Test
- **Paired Binary Data**: Tests for change in proportion
- **Test Statistic**: (|b - c| - 1)²/(b + c) where b,c are off-diagonal counts

## Statistical Power and Sample Size

### Power Analysis

#### Factors Affecting Power
1. **Effect Size**: Larger effects easier to detect
2. **Sample Size**: Larger samples increase power
3. **Significance Level**: Higher α increases power
4. **Variability**: Lower variability increases power

#### Effect Size Measures
- **Cohen's d**: (μ₁ - μ₂)/σ for two-sample t-test
- **Pearson r**: Correlation coefficient
- **Cohen's f**: For ANOVA
- **Odds Ratio**: For categorical data

#### Power Curves
Graphs showing power as function of:
- Effect size
- Sample size
- Significance level

### Sample Size Calculations

#### One-Sample t-test
n = (z_(α/2) + z_β)² × σ²/δ²
where δ = |μ - μ₀| is the effect size

#### Two-Sample t-test
n = 2(z_(α/2) + z_β)² × σ²/δ²
where δ = |μ₁ - μ₂|

#### Proportion Tests
n = (z_(α/2)√[p₀(1-p₀)] + z_β√[p₁(1-p₁)])²/(p₁ - p₀)²

#### Equivalence and Non-inferiority Tests
- **Different Formulas**: Because we're testing for similarity rather than difference
- **Margin of Equivalence**: Defines practically equivalent difference

### Post-hoc Power Analysis

#### Controversy
- **Observed Power**: Power calculated using observed effect size
- **Issues**: 
  - Circular reasoning
  - Doesn't provide useful information
- **Alternatives**: Confidence intervals, effect size estimation

#### Recommendations
- Plan power analysis before data collection
- Use confidence intervals for interpretation
- Consider practical significance

## Nonparametric Methods

### When to Use Nonparametric Tests

#### Advantages
- **Distribution-free**: Don't assume specific distributions
- **Robust**: Less sensitive to outliers
- **Ordinal Data**: Can handle ranked data

#### Disadvantages
- **Lower Power**: When parametric assumptions are met
- **Limited**: Fewer available procedures
- **Interpretation**: Sometimes less intuitive

### Rank-Based Tests

#### Wilcoxon Signed-Rank Test
- **Purpose**: One-sample or paired data test
- **Alternative to**: One-sample t-test
- **Assumptions**: Symmetry around median
- **Test Statistic**: Sum of signed ranks

#### Mann-Whitney U Test (Wilcoxon Rank-Sum)
- **Purpose**: Two independent samples
- **Alternative to**: Two-sample t-test
- **Assumptions**: Same shape distributions
- **Test Statistic**: U = n₁n₂ + n₁(n₁+1)/2 - R₁

#### Kruskal-Wallis Test
- **Purpose**: Multiple independent samples
- **Alternative to**: One-way ANOVA
- **Test Statistic**: H = (12/N(N+1))Σnᵢ(R̄ᵢ - R̄)²

#### Friedman Test
- **Purpose**: Multiple related samples
- **Alternative to**: Repeated measures ANOVA
- **Test Statistic**: Based on rank sums within blocks

### Distribution-Free Methods

#### Sign Test
- **Purpose**: Test median
- **Test Statistic**: Number of observations above hypothesized median
- **Distribution**: Binomial(n, 0.5) under H₀

#### Runs Test
- **Purpose**: Test randomness
- **Test Statistic**: Number of runs (sequences of same values)
- **Distribution**: Special tables or normal approximation

#### Spearman Rank Correlation
- **Purpose**: Test association without assuming linearity
- **Test Statistic**: r_s = 1 - 6Σd²ᵢ/[n(n²-1)]
- **Distribution**: Special tables or t-distribution approximation

#### Kendall's Tau
- **Purpose**: Alternative rank correlation
- **More Robust**: To outliers than Spearman
- **Interpretation**: Related to probability of concordance

### Bootstrap and Permutation Tests

#### Bootstrap Tests
1. **Resample**: With replacement from original sample
2. **Calculate**: Test statistic for each bootstrap sample
3. **P-value**: Proportion of bootstrap statistics more extreme than observed

#### Permutation Tests
1. **Permute**: Labels/group assignments
2. **Calculate**: Test statistic for each permutation
3. **P-value**: Proportion of permuted statistics more extreme than observed

#### Advantages
- **Exact**: P-values under null hypothesis
- **Flexible**: Can be applied to almost any statistic
- **No Assumptions**: About underlying distributions

## Bayesian Inference

### Bayesian Paradigm

#### Bayes' Theorem for Parameters
π(θ|x) = L(x|θ)π(θ) / ∫L(x|θ)π(θ)dθ

Where:
- π(θ): Prior distribution
- L(x|θ): Likelihood
- π(θ|x): Posterior distribution

#### Philosophical Differences
- **Frequentist**: Parameters are fixed, data is random
- **Bayesian**: Parameters are random, data is observed
- **Probability**: Bayesian allows probability statements about parameters

### Prior Distributions

#### Types of Priors
- **Informative**: Reflects substantial prior knowledge
- **Non-informative**: Minimal impact on posterior
- **Conjugate**: Posterior same family as prior
- **Improper**: Don't integrate to 1 (often non-informative)

#### Common Priors
- **Normal**: For location parameters
- **Gamma**: For precision/rate parameters
- **Beta**: For probability parameters
- **Uniform**: Often used as "non-informative"

#### Jeffreys' Prior
- **Invariant**: Under reparameterization
- **Formula**: π(θ) ∝ √|I(θ)| where I(θ) is Fisher information

### Conjugate Families

#### Normal-Normal
- **Prior**: μ ~ N(μ₀, σ₀²)
- **Likelihood**: X|μ ~ N(μ, σ²) (σ² known)
- **Posterior**: μ|x ~ N(μₙ, σₙ²)
- **Updates**: Precision-weighted average of prior and data

#### Beta-Binomial
- **Prior**: θ ~ Beta(α, β)
- **Likelihood**: X|θ ~ Binomial(n, θ)
- **Posterior**: θ|x ~ Beta(α + x, β + n - x)

#### Gamma-Poisson
- **Prior**: λ ~ Gamma(α, β)
- **Likelihood**: X|λ ~ Poisson(λ)
- **Posterior**: λ|x ~ Gamma(α + Σxᵢ, β + n)

### Bayesian Computation

#### Analytical Solutions
- **Conjugate Priors**: Closed-form posteriors
- **Simple Cases**: Few parameters, standard distributions

#### Markov Chain Monte Carlo (MCMC)
- **Metropolis-Hastings**: General MCMC algorithm
- **Gibbs Sampling**: For multivariate problems
- **Hamiltonian Monte Carlo**: Uses gradient information

#### Variational Inference
- **Approximation**: Replace intractable posterior with tractable family
- **Optimization**: Minimize KL divergence
- **Faster**: Than MCMC but approximate

### Bayesian Model Comparison

#### Marginal Likelihood
p(x) = ∫L(x|θ)π(θ)dθ

#### Bayes Factors
BF₁₂ = p(x|M₁)/p(x|M₂)

#### Information Criteria
- **DIC**: Deviance Information Criterion
- **WAIC**: Watanabe-Akaike Information Criterion
- **LOOIC**: Leave-One-Out Information Criterion

### Credible Intervals

#### Definition
Interval containing θ with probability 1-α under posterior distribution

#### Types
- **Equal-tailed**: α/2 probability in each tail
- **Highest Posterior Density (HPD)**: Shortest interval with 1-α probability

#### Comparison with Confidence Intervals
- **Interpretation**: Direct probability statements about parameter
- **Construction**: From posterior distribution rather than sampling distribution

## Multiple Comparisons and False Discovery Rate

### Multiple Testing Problem

#### Family-wise Error Rate (FWER)
Probability of making at least one Type I error among all tests

#### False Discovery Rate (FDR)
Expected proportion of false discoveries among all discoveries

#### Why It Matters
- **Increased Error**: Multiple tests inflate overall error rate
- **Reproducibility**: Important for scientific validity

### Classical Multiple Comparison Procedures

#### Bonferroni Correction
- **Method**: Use α/m for each test where m is number of tests
- **Controls**: FWER ≤ α
- **Conservative**: Especially for large m or dependent tests

#### Holm's Method
- **Step-down**: Order p-values, apply sequential Bonferroni
- **Less Conservative**: Than Bonferroni while controlling FWER
- **Procedure**: Reject H₀₍ᵢ₎ if p₍ᵢ₎ ≤ α/(m-i+1)

#### Sidak Correction
- **Formula**: 1 - (1-α)^(1/m) for each test
- **Assumes**: Independence of tests
- **Less Conservative**: Than Bonferroni under independence

### False Discovery Rate Methods

#### Benjamini-Hochberg Procedure
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
2. Find largest i such that p₍ᵢ₎ ≤ (i/m)α
3. Reject H₍₁₎, ..., H₍ᵢ₎

#### Benjamini-Yekutieli Procedure
- **For Dependent Tests**: Uses c(m) = Σ(1/j) from j=1 to m
- **More Conservative**: But valid under dependence

#### Storey's q-value
- **Estimates**: Positive false discovery rate (pFDR)
- **Adaptive**: Estimates proportion of true nulls
- **Software**: Available in R (qvalue package)

### Applications in Genomics and High-dimensional Data

#### Microarray Analysis
- **Thousands of Tests**: One for each gene
- **FDR Control**: Essential for biological interpretation

#### Genome-wide Association Studies (GWAS)
- **Millions of Tests**: One for each SNP
- **Stringent Thresholds**: Often 5×10⁻⁸

#### Proteomics and Metabolomics
- **Similar Issues**: Multiple testing across molecules
- **Pathway Analysis**: Groups related tests

### Practical Considerations

#### Dependence Structure
- **Positive Dependence**: FDR procedures often still valid
- **Arbitrary Dependence**: May need more conservative procedures

#### Prior Information
- **Weighted Procedures**: Give different weights to different tests
- **Hierarchical Testing**: Test groups first, then individuals

#### Reporting
- **Adjusted P-values**: vs. Discovery threshold
- **Confidence Intervals**: Should also be adjusted

## Advanced Topics

### Empirical Bayes Methods

#### Concept
- **Estimate Prior**: From data rather than specify subjectively
- **Shrinkage**: Individual estimates toward overall mean
- **Applications**: Multiple testing, small area estimation

#### James-Stein Estimator
- **Shrinkage**: For normal means
- **Paradox**: Dominates MLE for p ≥ 3 dimensions
- **Formula**: θ̂ᴶˢ = (1 - (p-2)σ²/||X||²)X

#### Empirical Bayes Testing
- **Two-component Model**: Mixture of null and alternative
- **Local FDR**: Probability that specific test is null

### Robust Statistical Methods

#### Types of Robustness
- **Outlier Resistance**: Not affected by extreme values
- **Distributional Robustness**: Valid under model deviations
- **Efficiency Robustness**: Good performance across conditions

#### Robust Location Estimators
- **Median**: 50% breakdown point
- **Trimmed Mean**: Remove extreme percentiles
- **M-estimators**: Minimize robust loss function
- **Huber Estimator**: Compromise between mean and median

#### Robust Scale Estimators
- **Median Absolute Deviation (MAD)**: median|xᵢ - median(x)|
- **Interquartile Range**: Q₃ - Q₁
- **Robust**: To outliers

#### Influence Function
- **Measures**: Effect of infinitesimal contamination
- **Bounded**: For robust estimators
- **Unbounded**: For non-robust estimators like mean

### Sequential Analysis

#### Sequential Probability Ratio Test (SPRT)
- **Continues**: Until sufficient evidence
- **Optimal**: Minimizes expected sample size
- **Boundaries**: Based on likelihood ratios

#### Group Sequential Methods
- **Clinical Trials**: Interim analyses at planned times
- **Alpha Spending**: Controls overall Type I error
- **Stopping Rules**: For efficacy or futility

#### Adaptive Designs
- **Modify**: Trial based on interim results
- **Sample Size**: Re-estimation
- **Population**: Enrichment

### Survival Analysis

#### Censoring
- **Right Censoring**: Know survival time ≥ observed time
- **Left Censoring**: Know survival time ≤ observed time
- **Interval Censoring**: Know survival time in interval

#### Kaplan-Meier Estimator
- **Non-parametric**: Estimate of survival function
- **Formula**: Ŝ(t) = ∏(tᵢ≤t) (1 - dᵢ/nᵢ)
- **Standard Error**: Greenwood's formula

#### Cox Proportional Hazards Model
- **Semi-parametric**: h(t|x) = h₀(t)exp(β'x)
- **Partial Likelihood**: Doesn't require baseline hazard specification
- **Assumptions**: Proportional hazards

#### Log-rank Test
- **Compare**: Survival curves
- **Non-parametric**: Doesn't assume specific distributions
- **Weights**: Equal across time

### Time Series Analysis

#### Stationarity
- **Weak Stationarity**: Constant mean, variance, autocovariance
- **Strong Stationarity**: Joint distributions invariant to shifts
- **Tests**: Augmented Dickey-Fuller, KPSS

#### Autoregressive Models (AR)
- **AR(p)**: Xₜ = φ₁Xₜ₋₁ + ... + φₚXₜ₋ₚ + εₜ
- **Stationarity**: Roots of characteristic equation outside unit circle

#### Moving Average Models (MA)
- **MA(q)**: Xₜ = εₜ + θ₁εₜ₋₁ + ... + θᵧεₜ₋ᵧ
- **Invertibility**: Roots of characteristic equation outside unit circle

#### ARIMA Models
- **Integration**: Differencing for non-stationary series
- **ARIMA(p,d,q)**: Combines AR, differencing, and MA
- **Box-Jenkins**: Methodology for model selection

## Practical Guidelines

### Choosing Statistical Tests

#### Data Type Considerations
- **Continuous vs. Categorical**: Determines available methods
- **Sample Size**: Affects choice between exact and approximate methods
- **Independence**: Paired vs. unpaired procedures

#### Assumption Checking
- **Normality**: Q-Q plots, Shapiro-Wilk test, histograms
- **Equal Variances**: F-test, Levene's test, plots
- **Independence**: Residual plots, Durbin-Watson test

#### Decision Trees
1. **What Type of Data?**: Continuous, ordinal, nominal
2. **How Many Groups?**: One, two, more than two
3. **Independent or Paired?**: Study design consideration
4. **Assumptions Met?**: Parametric vs. non-parametric

### Effect Size and Practical Significance

#### Why Effect Size Matters
- **Statistical Significance**: Can occur with large samples even for tiny effects
- **Practical Importance**: Requires consideration of effect magnitude
- **Meta-analysis**: Requires effect sizes for combination

#### Cohen's Guidelines
- **Small**: d = 0.2, r = 0.1, f = 0.1
- **Medium**: d = 0.5, r = 0.3, f = 0.25
- **Large**: d = 0.8, r = 0.5, f = 0.4

#### Confidence Intervals for Effect Sizes
- **Recommended**: Along with point estimates
- **Interpretation**: Range of plausible values
- **Planning**: Future studies based on intervals

### Reporting Statistical Results

#### Essential Elements
- **Descriptive Statistics**: Means, standard deviations, sample sizes
- **Test Statistics**: Value, degrees of freedom, p-value
- **Effect Sizes**: With confidence intervals
- **Assumptions**: How checked and any violations

#### P-value Reporting
- **Exact Values**: Rather than inequalities when possible
- **Context**: Statistical vs. practical significance
- **Multiple Testing**: Adjustments when applicable

#### Reproducibility
- **Data**: Make available when possible
- **Code**: Document analysis procedures
- **Methods**: Sufficient detail for replication

### Common Mistakes and How to Avoid Them

#### P-hacking
- **Definition**: Manipulating analysis to achieve significance
- **Prevention**: Pre-specify analysis plan, report all analyses
- **Transparency**: Document deviations from plan

#### Misinterpretation of P-values
- **Not**: Probability that null hypothesis is true
- **Not**: Probability of replication
- **Is**: Probability of observed or more extreme data given null

#### Confusing Statistical and Practical Significance
- **Large Samples**: Can detect trivial differences
- **Small Samples**: May miss important effects
- **Solution**: Always consider effect sizes and confidence intervals

#### Multiple Testing Issues
- **Problem**: Inflation of Type I error
- **When to Adjust**: Family of related hypotheses
- **Methods**: Choose appropriate for your context

#### Assumptions Violations
- **Check**: Before applying tests
- **Robust Methods**: When assumptions violated
- **Transformation**: Sometimes helps meet assumptions

### Software Considerations

#### Statistical Packages
- **R**: Free, extensive packages, programming required
- **SAS**: Commercial, comprehensive, good documentation
- **SPSS**: User-friendly interface, limited advanced methods
- **Stata**: Good for econometrics and biostatistics
- **Python**: Growing statistical capabilities, general programming

#### Reproducible Research
- **Version Control**: Git for code and analysis
- **Documentation**: R Markdown, Jupyter notebooks
- **Environment**: Docker, renv for R, virtual environments for Python
- **Data**: Document sources, cleaning procedures

#### Quality Control
- **Validation**: Compare results across packages when possible
- **Simulation**: Check procedures with known answers
- **Peer Review**: Have others check critical analyses

This comprehensive guide provides both theoretical foundations and practical guidance for statistical inference. Regular practice with real data and continued study of new developments in the field will help build expertise in applied statistics. 