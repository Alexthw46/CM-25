from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(results, x_key='lambda', y_key='lr', z_key='full_mean', title=None):
    """
    Plot a 3D surface from aggregated grid sweep results.

    :param results: List of dicts as returned by `gradient_grid_sweep`
    :param x_key: Key for x-axis (usually 'lambda')
    :param y_key: Key for y-axis (usually 'lr')
    :param z_key: Key for z-axis (e.g., 'full_mean', 'obs_mean')
    :param title: Optional plot title
    """

    # Extract unique values
    x_vals = sorted(set(res[x_key] for res in results))
    y_vals = sorted(set(res[y_key] for res in results))

    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Fill Z values
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            for res in results:
                if np.isclose(res[x_key], x) and np.isclose(res[y_key], y):
                    Z[i, j] = res[z_key]
                    break

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(np.log10(X), np.log10(Y), Z, cmap='viridis', edgecolor='k', alpha=0.9)
    ax.set_xlabel(f'log10({x_key})')
    ax.set_ylabel(f'log10({y_key})')
    ax.set_zlabel(z_key.replace('_', ' ').title())
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{z_key.replace('_', ' ').title()} over log-scaled grid")

    plt.tight_layout()
    plt.show()


def plot_lambda_results(aggregated_results):
    lams = [r['lambda'] for r in aggregated_results]
    obs_means = [r['obs_mean'] for r in aggregated_results]
    obs_stds = [r['obs_std'] for r in aggregated_results]
    full_means = [r['full_mean'] for r in aggregated_results]
    full_stds = [r['full_std'] for r in aggregated_results]
    iters_means = [r['iters_mean'] for r in aggregated_results]
    iters_stds = [r['iters_std'] for r in aggregated_results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.errorbar(lams, obs_means, yerr=obs_stds, fmt='o-', label="Observed error")
    plt.errorbar(lams, full_means, yerr=full_stds, fmt='x--', label="Full error")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lambda")
    plt.ylabel("Error")
    plt.title("Error vs Lambda (mean ± std)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.errorbar(lams, iters_means, yerr=iters_stds, fmt='s-', color='purple')
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("Iterations")
    plt.title("Iterations vs Lambda (mean ± std)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_unaggregated_per_seed(results_per_lambda):
    """
    Plot full and observed errors for each lambda, showing every seed's result (no aggregation).
    """
    lambda_values = sorted(results_per_lambda.keys())
    num_lambdas = len(lambda_values)

    plt.figure(figsize=(12, 5))

    # Full errors per seed
    plt.subplot(1, 2, 1)
    for lam in lambda_values:
        full_errs = results_per_lambda[lam]['full_errors']
        plt.plot([lam] * len(full_errs), full_errs, 'o', alpha=0.6, label=f'{lam:.1e}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lambda")
    plt.ylabel("Full Error (per seed)")
    plt.title("Full Error per Seed")
    plt.grid(True)

    # Observed errors per seed
    plt.subplot(1, 2, 2)
    for lam in lambda_values:
        obs_errs = results_per_lambda[lam]['obs_errors']
        plt.plot([lam] * len(obs_errs), obs_errs, 'x', alpha=0.6, label=f'{lam:.1e}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lambda")
    plt.ylabel("Observed Error (per seed)")
    plt.title("Observed Error per Seed")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def top_k_results(aggregated_results, k=5, sort_by='full_mean'):
    """
    Return the top-k parameter combinations from the aggregated results,
    sorted by the specified metric (ascending).

    :param aggregated_results: List of dicts as returned by the grid sweep
    :param k: Number of top results to return
    :param sort_by: Key to sort by (e.g., 'full_mean', 'obs_mean')
    :return: List of top-k results
    """
    sorted_results = sorted(aggregated_results, key=lambda r: r[sort_by])
    return sorted_results[:k]
