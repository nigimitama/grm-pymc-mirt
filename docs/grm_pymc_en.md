# Graded Response Model with PyMC

## What is GRM?

The **Graded Response Model (GRM)** (Samejima, 1969) is an Item Response Theory (IRT) model for ordered polytomous responses (e.g., Likert-scale items). It models the probability that a respondent's answer meets or exceeds category $k$:

$$
P(u_{ij} \geq k \mid \theta_i, a_j, b_{jk}) = \sigma\!\left(a_j(\theta_i - b_{jk})\right)
$$

| Parameter | Description |
|-----------|-------------|
| $\theta_i$ | Latent trait (ability/attitude) of respondent $i$ |
| $a_j$ | Discrimination parameter of item $j$ (slope) |
| $b_{jk}$ | Threshold parameter for category $k$ of item $j$, ordered: $b_{j1} < b_{j2} < \cdots$ |

The category response probability is then:

$$
P(u_{ij} = k) = P(u_{ij} \geq k) - P(u_{ij} \geq k+1)
$$

with $P(u_{ij} \geq 0) = 1$ and $P(u_{ij} \geq K_j) = 0$.

## PyMC Implementation

### Priors

```python
theta ~ Normal(0, 1)          # respondent latent trait
a     ~ LogNormal(0, 0.5)     # item discrimination (positive)
b     ~ Normal(0, 1.5)        # item thresholds (ordered constraint)
```

### Model definition

```python
import pymc as pm
import numpy as np

with pm.Model() as grm:
    theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=N)
    a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=J)
    b = pm.Normal(
        "b",
        mu=0.0,
        sigma=1.5,
        transform=pm.distributions.transforms.ordered,
        initval=np.tile(np.linspace(-1.0, 1.0, K - 1), (J, 1)),
        shape=(J, K - 1),
    )
    # cutpoints: c_{jk} = a_j * b_{jk}
    c = pm.Deterministic("cutpoints", a[:, None] * b)

    # linear predictor: eta_{ij} = a_j * theta_i
    person_idx = np.repeat(np.arange(N), J)
    item_idx   = np.tile(np.arange(J), N)
    eta = a[item_idx] * theta[person_idx]

    pm.OrderedLogistic(
        "Y_obs",
        eta=eta,
        cutpoints=c[item_idx],
        observed=Y.ravel(),
        compute_p=False,
    )
```

> **Note:** The `ordered` transform ensures $b_{j1} < b_{j2} < \cdots < b_{j,K-1}$ automatically.

### Sampling

```python
with grm:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        nuts_sampler="numpyro",  # faster on GPU/CPU via JAX
    )
```

### Handling varying category counts

When items have different numbers of categories, group items by their category count and define separate `b` variables for each group:

```python
for k in unique_K:
    item_idx_k = np.where(K_per_item == k)[0]
    b_k = pm.Normal(
        f"b_{k}",
        mu=0.0, sigma=1.5,
        shape=(len(item_idx_k), k - 1),
        transform=pm.distributions.transforms.ordered,
        initval=np.tile(np.linspace(-1.0, 1.0, k - 1), (len(item_idx_k), 1)),
    )
    ...
    pm.OrderedLogistic(f"y_{k}", eta=eta_k, cutpoints=cutpoints_k, observed=Y[:, item_idx_k])
```

## Validation

After sampling, verify convergence with $\hat{R}$ (should be $\leq 1.01$):

```python
import arviz as az

summary = az.summary(idata, var_names=["a", "b", "theta"])
print(summary.query("r_hat > 1.01"))  # should be empty
```

## Datasets

| Directory | Dataset | N | J | K |
|-----------|---------|---|---|---|
| [`simulate_dataset/`](../simulate_dataset/) | Synthetic data | 1 000 | 20 | 3–5 (mixed) |
| [`RWAS_dataset/`](../RWAS_dataset/) | Right-Wing Authoritarianism Scale | 9 680 | 22 | 9 |

## References

- Samejima, F. (1969). *Estimation of latent ability using a response pattern of graded scores.* Psychometrika Monograph, 17.
- Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R Environment. *Journal of Statistical Software*, 48(6), 1–29.
- PyMC documentation: [Ordinal Regression](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-ordinal-regression.html)
