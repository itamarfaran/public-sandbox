# `statistical.py`: minimum detectable effects for A/B and many-to-one tests

`statistical.py` answers one question: **given a sample size, how large must the true effect be
for the experiment to detect it?** That threshold is the *minimum detectable effect* (MDE).

The module covers two kinds of metric:

| Metric | Function | Returns |
| --- | --- | --- |
| Conversion rate (binary outcome) | `calculate_p_B` | the treatment rate $p_B$ at the MDE |
| Continuous mean | `calculate_mu_B` | the treatment mean $\mu_B$ at the MDE |

Both functions handle two designs:

- a classic **A/B test** (one control, one treatment), and
- a **many-to-one test** (one control, $k > 1$ treatments, each compared with the control), where
  the significance threshold is corrected for multiple comparisons using **Dunnett's method**.

The functions return the *treatment value*, not the difference. The absolute MDE is
`p_B - p` (or `mu_B - mu`). The relative MDE is `p_B / p - 1`.

---

## Shared parameters

| Parameter | Meaning |
| --- | --- |
| `n` | Total sample size across **all** groups. |
| `xi` | How `n` is split between groups (see below). |
| `alpha` | Family-wise significance level, two-sided. When $k > 1$ it is Dunnett-adjusted, so the chance of *any* false positive across all treatment arms is `alpha`. |
| `power` | Target power **per treatment arm**: the probability that a given arm whose true effect equals the MDE comes out significant. |

`xi` can be given in two forms:

- **A float**, for an A/B test: the fraction of `n` that goes to treatment, so
  $n_B = n \cdot \xi$ and $n_A = n(1-\xi)$. `xi=0.5` is an even split.
- **A sequence**, for any number of groups: all group sizes or weights, **control first**,
  e.g. `[400, 200, 200, 200]` for a control twice the size of each of three treatments. The values
  are normalized, so only their ratios matter. Their scale does not.

When the treatment groups are unequal, the functions return the MDE of the **smallest** treatment
group. That group has the least data and so the largest MDE, which makes it the worst case.

---

## The core formula

Every function here is built on the standard large-sample power equation for a two-sample z-test:

$$
\Delta_{\text{MDE}} = (d + z_{1-\beta}) \cdot SE
$$

where

- $d$ is the critical value of the test: $z_{1-\alpha/2} \approx 1.96$ for a plain A/B test at
  $\alpha = 0.05$, or the Dunnett critical value when there are several treatments,
- $z_{1-\beta}$ is the normal quantile of the target power ($\approx 0.84$ for 80% power), and
- $SE$ is the standard error of the difference between the treatment and control estimates.

In the code, $d + z_{1-\beta}$ is `z_sum`. As usual, the formula ignores the negligible chance of
rejecting in the wrong tail.

### Helper: `_d_and_xi(xi, alpha)`

This helper turns `xi` into the three numbers the formulas need:

- `d`: the critical value, obtained from `dunnett_d` (for a float `xi` this is just
  $z_{1-\alpha/2}$, because Dunnett with one treatment is the ordinary z-test),
- `xi0`: the control's fraction of `n`,
- `xi_t`: the treatment's fraction of `n` (the **smallest** treatment's fraction when there are
  several).

---

## `calculate_mu_B`: continuous metrics

```python
calculate_mu_B(n, mu, sigma, xi, alpha, power) -> float
```

This assumes every group shares a common standard deviation `sigma`. The standard error of the
difference is then known up front:

$$
SE = \sigma \sqrt{\frac{1}{n\,\xi_t} + \frac{1}{n\,\xi_0}}
$$

so the answer is a closed-form expression:

$$
\mu_B = \mu + (d + z_{1-\beta}) \cdot SE
$$

**Caveat:** this uses the normal approximation. For small groups, a t-based calculation is exact
and gives a slightly larger MDE, so this result is somewhat optimistic (anti-conservative) there.

---

## `calculate_p_B`: conversion rates

```python
calculate_p_B(n, p, xi, alpha, power) -> float
```

Proportions are harder because the variance depends on the rate itself. The standard error of
the difference uses the **unpooled** variance, so it depends on the unknown $p_B$:

$$
p_B - p = (d + z_{1-\beta}) \sqrt{\frac{p_B(1-p_B)}{n\,\xi_t} + \frac{p(1-p)}{n\,\xi_0}}
$$

Squaring both sides and multiplying through by $n\,\xi_t$ gives a quadratic
$a\,p_B^2 + b\,p_B + c = 0$. Writing $Z^2 = (d + z_{1-\beta})^2$:

$$
\begin{aligned}
a &= n\,\xi_t + Z^2 \\
b &= -\left(2\,n\,\xi_t\,p + Z^2\right) \\
c &= n\,\xi_t\,p^2 - Z^2\,\xi_t\,\frac{p(1-p)}{\xi_0}
\end{aligned}
$$

These are exactly `a`, `b` and `c` in the code. The function takes the **larger root**, which is
the upward effect ($p_B > p$).

Notes:

- The result matches a test that uses the **unpooled** standard error. A test that uses the pooled
  standard error under $H_0$ has slightly different power.
- If the root is above 1, no achievable conversion rate is detectable with this sample size, and
  the function returns `nan`.
- Only increases are computed. To get a downward MDE, use the lower root of the same quadratic.

---

## `dunnett_d`: the multiple-comparison critical value

```python
dunnett_d(*sizes, alpha=0.05, two_sided=True, eps=1e-5, seed=0) -> float
```

### Why it is needed

With $k$ treatments each compared with the control, testing every arm at $\alpha$ inflates the
chance of at least one false positive. A simple fix is **Bonferroni**: test each arm at
$\alpha / k$. That ignores the fact that the $k$ test statistics are **positively correlated**,
because every comparison reuses the same control group. Dunnett's method takes that correlation
into account and finds the smallest critical value $d$ that keeps the family-wise error at exactly
$\alpha$. The result always lies between the unadjusted and the Bonferroni critical values.

### How it works

1. **Correlation structure.** Under $H_0$, with $Z_i = (\bar X_i - \bar X_0) / SE_i$ and equal
   variances, the statistics are jointly normal with

   $$
   \operatorname{corr}(Z_i, Z_j) = \sqrt{\frac{n_i}{n_i + n_0}} \cdot \sqrt{\frac{n_j}{n_j + n_0}}
   $$

   The code builds this matrix as the outer product of `r = sqrt(ni / (ni + n0))`, with ones on
   the diagonal. A larger control lowers the correlation, which pushes $d$ toward the Bonferroni
   value.

2. **Find $d$.** The function solves

   $$
   P\left(|Z_1| < d, \dots, |Z_k| < d\right) = 1 - \alpha
   $$

   (or $P(Z_i < d \ \forall i) = 1 - \alpha$ when one-sided). It evaluates the multivariate normal
   CDF with SciPy and finds the root with Brent's method. The search bracket runs from the
   unadjusted value $z_{1-a}$ to the Bonferroni value $z_{1-a/k}$, where $a = \alpha/2$ for a
   two-sided test, each widened by 0.01.

3. **Shortcut.** With a single treatment ($k = 1$) the answer is just $z_{1-a}$.

### Parameters

| Parameter | Meaning |
| --- | --- |
| `*sizes` | Group sizes or fractions; the **first is the control**, the rest are treatments. |
| `alpha` | Family-wise significance level. |
| `two_sided` | Reject when $\lvert Z_i \rvert > d$ (`True`) or when $Z_i > d$ (`False`). |
| `eps` | Absolute and relative error tolerance of the multivariate normal CDF integration. |
| `seed` | Seed for the randomized CDF integration, so that repeated calls return the same value. `None` lets results vary by about `eps` from run to run. |

It raises `ValueError` if the sizes are not a flat sequence, if there are fewer than two groups,
or if any size is not positive.

**Caveats:**

- Like the rest of the module, this is the large-sample (known-variance) z version. It is slightly
  anti-conservative for small groups compared with the exact t-based Dunnett value.
- The correlation formula assumes equal variances across groups. For proportions, where variance
  depends on the rate, this is an approximation.

---

## Examples

```python
from statistical import calculate_p_B, calculate_mu_B, dunnett_d

# A/B test, 10,000 users split 50/50, 10% baseline conversion, alpha=0.05, 80% power
calculate_p_B(10_000, p=0.10, xi=0.5, alpha=0.05, power=0.8)
# -> 0.1174  (detects a lift from 10.0% to ~11.7%)

# Same budget, control + 3 treatments, control twice as large as each treatment
calculate_p_B(10_000, p=0.10, xi=[400, 200, 200, 200], alpha=0.05, power=0.8)
# -> 0.1285  (larger MDE: smaller arms plus the multiplicity correction)

# Continuous metric: baseline mean 100, sd 20
calculate_mu_B(10_000, mu=100, sigma=20, xi=0.5, alpha=0.05, power=0.8)
# -> 101.12
calculate_mu_B(10_000, mu=100, sigma=20, xi=[400, 200, 200, 200], alpha=0.05, power=0.8)
# -> 101.76

# Dunnett critical values, alpha=0.05 two-sided
dunnett_d(1, 1)        # -> 1.960  (single comparison: plain z)
dunnett_d(1, 1, 1, 1)  # -> 2.349  (3 equal treatments; Bonferroni would be 2.394)
dunnett_d(2, 1, 1, 1)  # -> 2.372  (bigger control -> lower correlation -> closer to Bonferroni)
```

## Dependencies

`numpy` and `scipy` (`scipy.stats` for the normal and multivariate normal distributions,
`scipy.optimize.brentq` for root finding).
