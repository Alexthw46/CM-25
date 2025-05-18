import matplotlib.pyplot as plt
import numpy as np


def plot_3d_surface(results, x_key='lambda', y_key='lr', title=None):
    """
    Plot 3D surfaces for both full and observed errors from aggregated grid sweep results.

    :param results: List of dicts as returned by `gradient_grid_sweep`
    :param x_key: Key for x-axis (usually 'lambda')
    :param y_key: Key for y-axis (usually 'lr')
    :param title: Optional plot title
    """

    # Extract unique values
    x_vals = sorted(set(res[x_key] for res in results))
    y_vals = sorted(set(res[y_key] for res in results))

    X, Y = np.meshgrid(x_vals, y_vals)
    Z_full = np.zeros_like(X)
    Z_obs = np.zeros_like(X)

    # Fill Z values for both full and observed errors
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            for res in results:
                if np.isclose(res[x_key], x) and np.isclose(res[y_key], y):
                    Z_full[i, j] = res.get('full_mean', np.nan)
                    Z_obs[i, j] = res.get('obs_mean', np.nan)
                    break

    fig = plt.figure(figsize=(18, 7))

    # Full error surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(np.log10(X), np.log10(Y), Z_full, cmap='viridis', edgecolor='k', alpha=0.9)
    ax1.set_xlabel(f'log10({x_key})')
    ax1.set_ylabel(f'log10({y_key})')
    ax1.set_zlabel('Full Error (mean)')
    ax1.set_title("Full Error (mean) over log-scaled grid")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Observed error surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(np.log10(X), np.log10(Y), Z_obs, cmap='plasma', edgecolor='k', alpha=0.9)
    ax2.set_xlabel(f'log10({x_key})')
    ax2.set_ylabel(f'log10({y_key})')
    ax2.set_zlabel('Observed Error (mean)')
    ax2.set_title("Observed Error (mean) over log-scaled grid")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    if title:
        fig.suptitle(title)

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
