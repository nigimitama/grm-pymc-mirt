# PyMC による段階反応モデル (GRM)

## GRM とは

**段階反応モデル (Graded Response Model; GRM)** (Samejima, 1969) は、リッカート尺度のような順序多値反応データに対する項目反応理論 (IRT) モデルです。カテゴリ $k$ 以上の反応をする累積確率を次のように定義します。

$$
P(u_{ij} \geq k \mid \theta_i, a_j, b_{jk}) = \sigma\!\left(a_j(\theta_i - b_{jk})\right)
$$

| パラメータ | 説明 |
|-----------|------|
| $\theta_i$ | 回答者 $i$ の潜在特性（能力・態度） |
| $a_j$ | 項目 $j$ の識別力（傾き、正値） |
| $b_{jk}$ | 項目 $j$ のカテゴリ $k$ に対する閾値。$b_{j1} < b_{j2} < \cdots$ の順序制約あり |

カテゴリ反応確率は以下で得られます。

$$
P(u_{ij} = k) = P(u_{ij} \geq k) - P(u_{ij} \geq k+1)
$$

ただし $P(u_{ij} \geq 0) = 1$、$P(u_{ij} \geq K_j) = 0$。

## PyMC による実装

### 事前分布

```python
theta ~ Normal(0, 1)          # 回答者の潜在特性
a     ~ LogNormal(0, 0.5)     # 識別力（正値を保証）
b     ~ Normal(0, 1.5)        # 閾値パラメータ（順序制約付き）
```

### モデル定義

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
    # カットポイント: c_{jk} = a_j * b_{jk}
    c = pm.Deterministic("cutpoints", a[:, None] * b)

    # 線形予測子: eta_{ij} = a_j * theta_i
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

> **ポイント:** `ordered` トランスフォームを指定することで、$b_{j1} < b_{j2} < \cdots < b_{j,K-1}$ の順序制約を自動的に満たします。

### サンプリング

```python
with grm:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        nuts_sampler="numpyro",  # JAX 経由で高速化
    )
```

### カテゴリ数が項目ごとに異なる場合

カテゴリ数 $K_j$ が項目によって異なるときは、カテゴリ数ごとに項目をグループ化して別々の `b` 変数を定義します。

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

## 収束診断

サンプリング後は $\hat{R}$（1.01 以下が目安）で収束を確認します。

```python
import arviz as az

summary = az.summary(idata, var_names=["a", "b", "theta"])
print(summary.query("r_hat > 1.01"))  # 空であれば収束している
```

## データセット

| ディレクトリ | データセット | N | J | K |
|------------|-----------|---|---|---|
| [`simulate_dataset/`](../simulate_dataset/) | シミュレーションデータ | 1,000 | 20 | 3〜5（混合） |
| [`RWAS_dataset/`](../RWAS_dataset/) | 右翼権威主義尺度 (RWAS) | 9,680 | 22 | 9 |

## 参考文献

- Samejima, F. (1969). *Estimation of latent ability using a response pattern of graded scores.* Psychometrika Monograph, 17.
- Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R Environment. *Journal of Statistical Software*, 48(6), 1–29.
- PyMC ドキュメント: [Ordinal Regression](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-ordinal-regression.html)
