import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from solvers import *
import numpy as np
import time

def compare_solvers(X_obs, X_true, u0, v0, mask, lambda_als=None, gd_params=None, plot=False, verbose=True, patience=10,
                    max_it=5000):
    """
    Compare the performance of Alternating Optimization (ALS), ALS with normalization and Gradient Descent.

    :param max_it:
    :param X_obs: Incomplete matrix (n x n), missing entries as zeros
    :param X_true: Ground truth matrix (n x n)
    :param u0: Initial guess for vector u (n,).
    :param v0: Initial guess for vector v (n,).
    :param mask: Binary mask of observed entries (n x n), dtype=bool
    :param lambda_als: Regularization strength for ALS and NormALS as a list
    :param gd_params: Gradient descent parameters as a list [lr, lambda_reg]
    :param plot: If True, plot residual curves
    """

    if lambda_als is None:
        lambda_als = [0.403, 1.5e-2]

    if verbose:
        print("=== Alternating Optimization (ALS) ===")
    start = time.time()
    u, v, it_als, res, hist = alternating_optimization(
        X_obs, mask, u0.copy(), v=v0.copy(), max_it=max_it,
        lambda_reg=lambda_als[0], verbose=False, track_residuals=plot,
        eps=1e-8, patience=patience
    )
    end = time.time()
    time_als = end - start
    ALS_sol = np.outer(u, v)
    observed_error_ALS = np.linalg.norm((ALS_sol - X_true) * mask, 'fro')
    full_error_ALS = np.linalg.norm(ALS_sol - X_true, 'fro')
    if verbose:
        print(
            f"ALS: Residual={res:.6f}, Observed Error={observed_error_ALS:.8f}, Full Error={full_error_ALS:.8f}, Iter={it_als}, Time={end - start:.4f}s")

    if verbose:
        print("\n=== ALS with Normalization (NormALS) ===")
    start = time.time()
    u, v, it_NormAls, res, hist2 = alternating_optimization(
        X_obs, mask, u0.copy(), v=v0.copy(), max_it=max_it,
        lambda_reg=lambda_als[1], norm_v=True, verbose=False, track_residuals=plot,
        eps=1e-8, patience=patience

    )
    end = time.time()
    time_NormALS = end - start
    NormALS_sol = np.outer(u, v)
    observed_error_NormALS = np.linalg.norm((NormALS_sol - X_true) * mask, 'fro')
    full_error_NormALS = np.linalg.norm(NormALS_sol - X_true, 'fro')
    if verbose:
        print(
            f"NormALS: Residual={res:.6f}, Observed Error={observed_error_NormALS:.8f}, Full Error={full_error_NormALS:.8f}, Iter={it_NormAls}, Time={end - start:.4f}s")

    if verbose:
        print("\n=== Gradient Descent (GD) ===")
    if gd_params is None:
        gd_params = [9.41e-04, 6.95e-05]
    lr = gd_params[0]
    lambda_gd = gd_params[1]
    start = time.time()
    u, v, it_gd, res, hist3 = gradient_descent_rank1(
        X_obs, mask, u_init=u0.copy(), v_init=v0.copy(),
        max_it=max_it, lr=lr, lambda_reg=lambda_gd,
        tol=1e-8, verbose=False, track_residuals=plot, patience=patience,
        gradient_clip=10.0
    )
    end = time.time()
    time_gd = end - start
    gd_sol = np.outer(u, v)
    observed_error_gd = np.linalg.norm((gd_sol - X_true) * mask, 'fro')
    full_error_gd = np.linalg.norm(gd_sol - X_true, 'fro')
    if verbose:
        print(
            f"GD: Residual={res:.6f}, Observed Error={observed_error_gd:.8f}, Full Error={full_error_gd:.8f}, Iter={it_gd}, Time={end - start:.4f}s")

    # Plot residuals if available
    if plot:
        plt.figure(figsize=(10, 6))
        if hist and hist['objective']:
            plt.plot(hist['objective'], label='ALS Observed Error')
        if hist2 and hist2['objective']:
            plt.plot(hist2['objective'], label='NormALS Observed Error')
        if hist3:
            plt.plot(hist3, label='GD Observed Error')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Frobenius Error on Observed Entries')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot the residuals
        plt.figure(figsize=(10, 6))
        if hist and hist['residuals']:
            plt.plot(hist['residuals'], label='ALS Residuals')
        if hist2 and hist2['residuals']:
            plt.plot(hist2['residuals'], label='NormALS Residuals')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Residuals Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # group the results in a dictionary
    results = {
        'ALS': {
            'observed_error': observed_error_ALS,
            'full_error': full_error_ALS,
            'time': time_als,
            'iterations': it_als,
        },
        'NormALS': {
            'observed_error': observed_error_NormALS,
            'full_error': full_error_NormALS,
            'time': time_NormALS,
            'iterations': it_NormAls,
        },
        'GD': {
            'observed_error': observed_error_gd,
            'full_error': full_error_gd,
            'time': time_gd,
            'iterations': it_gd,
        }
    }
    return results


def run_benchmark_for_seed(seed, init_settings, density=0.1, m=100, n=100, noise=0.0, snr=None):
    """
    Run the benchmark for a given seed over the different solvers and initialization strategies.
    :param init_settings: Dictionary containing settings for each solver and initialization strategy.
    :param seed: Random seed for reproducibility
    :param density: Density of the observed entries in the matrix
    :param m: Number of rows in the matrix
    :param n: Number of columns in the matrix
    :param noise: Amount of noise to add to the observed entries as standard deviation
    :param snr: Signal-to-noise ratio
    :return: Dictionary containing the results for each method
    """

    X_true, X_obs, mask, u_true, v_true = generate_synthetic_problem(m, n, density, seed=seed - 1, noise_std=noise, snr=snr)
    results = {}

    svd_results = baseline_svd(X_true, X_obs, mask)
    results['Baseline SVD'] = {
        'Baseline SVD': {
            'observed_error': svd_results[0],
            'full_error': svd_results[1],
            'time': svd_results[2],
            'iterations': 1
        }
    }

    for strategy_key in ['gaussian', 'svd', 'svd+noise', 'mean']:
        pretty_name = strategy_key.replace('+', ' + ').capitalize()

        init_time = time.time()
        if strategy_key == 'svd+noise':
            u0, v0 = initialize_uv(X_obs, mask, strategy='svd', epsilon=1e-4, seed=seed)
        else:
            u0, v0 = initialize_uv(X_obs, mask, strategy=strategy_key, seed=seed)
        init_time = time.time() - init_time
        # Get strategy-specific parameters
        params = init_settings[strategy_key]
        lambda_als = params['lambda_als']
        gd_params = params['gd_params']

        results[pretty_name] = compare_solvers(X_obs, X_true, u0.copy(), v0.copy(), mask, lambda_als=lambda_als,
                                               gd_params=gd_params, plot=False, verbose=False, patience=3, max_it=5000)
        # Add the time needed for initialization
        for solver in results[pretty_name]:
            results[pretty_name][solver]['time'] += init_time
    return results


def summarize_solver_results(solver, accum_results):
    rows = []
    # Special case for Baseline SVD since it doesn't have an init method
    if solver == 'Baseline SVD':
        obs = np.mean(accum_results[solver][f'{solver}_obs'])
        full = np.mean(accum_results[solver][f'{solver}_full'])
        time = np.mean(accum_results[solver][f'{solver}_time'])
        rows.append({
            'Observed Error': obs,
            'Full Error': full,
            'Time': time
        })
        return pd.DataFrame(rows)
    for method in ['Gaussian', 'Svd', 'Svd + noise', 'Mean']:
        obs = np.mean(accum_results[method][f'{solver}_obs'])
        obs_err_std = np.std(accum_results[method][f'{solver}_obs'])
        full = np.mean(accum_results[method][f'{solver}_full'])
        full_err_std = np.std(accum_results[method][f'{solver}_full'])
        time = np.mean(accum_results[method][f'{solver}_time'])
        if f'{solver}_iterations' in accum_results[method]:
            iterations = np.mean(accum_results[method][f'{solver}_iterations'])
        else:
            iterations = 1  # Default to 1 if not available

        rows.append({
            'Method': method,
            'Observed Error': obs,
            'Observed Error Std': obs_err_std,
            'Full Error': full,
            'Full Error Std': full_err_std,
            'Time': time,
            'Iterations': iterations
        })
    return pd.DataFrame(rows)