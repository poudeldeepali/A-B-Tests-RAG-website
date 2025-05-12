# ab_test.py

import time
import random
import uuid

import numpy as np
from scipy import stats
from chromadb import Client
from models.vectordb_setup import vectordb

def fetch_response_times():
    """
    Fetch all entries from the Chroma 'metrics' collection
    and return two numpy arrays: production vs shadow.
    """
    client: Client = vectordb._client
    metrics = client.get_or_create_collection("metrics")
    docs = metrics.get()
    metadatas = docs.get("metadatas", [])

    prod = [m["response_time"] for m in metadatas if m.get("model") == "production"]
    shadow = [m["response_time"] for m in metadatas if m.get("model") == "shadow"]
    return np.array(prod), np.array(shadow)

def generate_synthetic_data(n_prod=800, n_shadow=200,
                            mu_prod=0.5, sigma_prod=0.1,
                            mu_shadow=0.6, sigma_shadow=0.15,
                            min_rt=0.1, max_rt=2.0, seed=42):
    """
    Generate synthetic response-time data for production and shadow groups,
    drawn from truncated normal distributions.
    """
    rng = np.random.default_rng(seed)

    def truncated_normal(size, mu, sigma):
        samples = rng.normal(mu, sigma, size*2)
        samples = samples[(samples >= min_rt) & (samples <= max_rt)]
        if len(samples) < size:
            # if too few, pad by resampling simply from mu
            samples = np.concatenate([samples, rng.normal(mu, sigma, size-len(samples))])
        return samples[:size]

    prod = truncated_normal(n_prod, mu_prod, sigma_prod)
    shadow = truncated_normal(n_shadow, mu_shadow, sigma_shadow)
    return prod, shadow

def t_test(prod, shadow):
    """Welch’s two-sample t-test."""
    return stats.ttest_ind(prod, shadow, equal_var=False)

def mann_whitney(prod, shadow):
    """Mann–Whitney U test (two-sided)."""
    return stats.mannwhitneyu(prod, shadow, alternative="two-sided")

def bootstrap_diff_of_means(prod, shadow, n_boot=10000, alpha=0.05, seed=42):
    """
    Bootstrap difference of means (prod - shadow) with CI.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s_p = rng.choice(prod, size=len(prod), replace=True)
        s_s = rng.choice(shadow, size=len(shadow), replace=True)
        diffs[i] = s_p.mean() - s_s.mean()

    lower = np.percentile(diffs, 100 * (alpha/2))
    upper = np.percentile(diffs, 100 * (1 - alpha/2))
    return diffs.mean(), (lower, upper)

def summarize(prod, shadow):
    print(f"Production: n={len(prod)}, mean={prod.mean():.3f}, median={np.median(prod):.3f}")
    print(f"Shadow    : n={len(shadow)}, mean={shadow.mean():.3f}, median={np.median(shadow):.3f}\n")

    t_stat, t_p = t_test(prod, shadow)
    print(f"Welch’s t-test       : t = {t_stat:.3f}, p = {t_p:.3e}")

    u_stat, u_p = mann_whitney(prod, shadow)
    print(f"Mann–Whitney U test  : U = {u_stat:.3f}, p = {u_p:.3e}")

    mean_diff, (ci_low, ci_high) = bootstrap_diff_of_means(prod, shadow)
    print(f"Bootstrap Δmean      : {mean_diff:.3f}")
    print(f"95% CI for Δmean     : [{ci_low:.3f}, {ci_high:.3f}]\n")

if __name__ == "__main__":
    prod_times, shadow_times = fetch_response_times()

    # If either group is empty, generate synthetic data
    if len(prod_times) == 0 or len(shadow_times) == 0:
        print("No real data found—generating synthetic samples...\n")
        prod_times, shadow_times = generate_synthetic_data(
            n_prod=800, n_shadow=200,
            mu_prod=0.5, sigma_prod=0.1,
            mu_shadow=0.6, sigma_shadow=0.15
        )

    summarize(prod_times, shadow_times)
