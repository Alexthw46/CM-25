# Press Maiusc+F10 to execute it or replace it with your code.
import time
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def alternating_optimization(u, X, X_mask, max_it=100, eps=1e-8, lambda_reg=1e-8, verbose=False, track_residuals=False, norm_v=False):
    """
    Alternating optimization for matrix completion.

    :param u: Initial guess for vector u (n,)
    :param X: Incomplete matrix (n x n), where missing entries are zeros
    :param X_mask: Binary mask of observed entries (n x n)
    :param max_it: Max number of iterations
    :param eps: Tolerance on improvement
    :param lambda_reg: Regularization strength (lambda)
    :param verbose: If True, print progress
    :param track_residuals: If True, track residuals
    :return:
        u, v: optimized vectors
        it: number of iterations
        res: final cost
        (optionally) residual_history: history of residuals
    """

    n = X.shape[0]
    v = np.zeros(n)
    it = 0
    prev_res = 1e8
    improvement = prev_res

    residual_history = []

    while it < max_it and improvement > eps:
        it += 1

        # Update v
        for j in range(n):
            col_mask = X_mask[:, j]
            if not np.any(col_mask):
                v[j] = 0
            else:
                a = X[col_mask, j] @ u[col_mask]
                b = u[col_mask]
                prod = b @ b + lambda_reg
                v[j] = a / prod

        # Update u
        for i in range(n):
            row_mask = X_mask[i, :]
            if not np.any(row_mask):
                u[i] = 0
            else:
                a = X[i, row_mask] @ v[row_mask]
                b = v[row_mask]
                prod = b @ b + lambda_reg
                u[i] = a / prod

        # Residual only on observed entries
        A_hat = np.outer(u, v)
        res = np.linalg.norm((A_hat - X) * X_mask, 'fro')
        improvement = prev_res - res
        prev_res = res

        # Normalize v and adjust u
        if norm_v:
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                v /= v_norm
                u *= v_norm


        if verbose and (it % 10 == 0 or it == 1):
            print(f"[AO] Iter {it}, Residual: {res:.6f}, Improvement: {improvement:.6f}")
        if track_residuals:
            residual_history.append(res)

    return u, v, it, res, residual_history if track_residuals else None


def multi_seed_lambda_sweep(num_seeds, n, density, lambda_values, maxit=500, eps=1e-6):
    """
    Perform tests on a set of lambda values across multiple seeds to assess average performance.

    :param num_seeds: Number of different random seeds to test
    :param n: Size of the square matrix
    :param density: Probability of observing each entry
    :param lambda_values: List of lambda values to test
    :param maxit: Max iterations for alternating optimization
    :param eps: Convergence tolerance
    :return: dict mapping lambda to aggregated results (mean/std of errors and iterations)
    """
    results_per_lambda = {lam: {'obs_errors': [], 'full_errors': [], 'iters': []} for lam in lambda_values}

    for seed in range(num_seeds):
        np.random.seed(seed)

        u_true = np.random.randn(n)
        v_true = np.random.randn(n)
        X_true = np.outer(u_true, v_true)

        mask = (np.random.rand(n, n) < density)
        X_obs = X_true * mask
        u0 = np.random.randn(n)

        print(f"\n=== Seed {seed} ===")
        for lam in lambda_values:
            u, v, it, res, _ = alternating_optimization(
                u0.copy(), X_obs, mask, max_it=maxit, eps=eps, lambda_reg=lam, verbose=False, track_residuals=False
            )
            A_hat = np.outer(u, v)

            observed_error = np.linalg.norm((A_hat - X_true) * mask, ord='fro')
            full_error = np.linalg.norm(A_hat - X_true, ord='fro')

            results_per_lambda[lam]['obs_errors'].append(observed_error)
            results_per_lambda[lam]['full_errors'].append(full_error)
            results_per_lambda[lam]['iters'].append(it)

            print(f"[lambda={lam:.2e}] obs_error={observed_error:.4f}, full_error={full_error:.4f}, iters={it}")

    # Compute mean and std per lambda
    aggregated_results = []
    for lam in lambda_values:
        res = results_per_lambda[lam]
        aggregated_results.append({
            'lambda': lam,
            'obs_mean': np.mean(res['obs_errors']),
            'obs_std': np.std(res['obs_errors']),
            'full_mean': np.mean(res['full_errors']),
            'full_std': np.std(res['full_errors']),
            'iters_mean': np.mean(res['iters']),
            'iters_std': np.std(res['iters']),
        })

    return aggregated_results, results_per_lambda


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


def soft_impute(X_obs, mask, rank=1, lambda_soft=1e-2, max_iters=100, tol=1e-4, verbose=False, constrained=True):
    """
    Soft-Impute with optional constraint on observed entries.

    :param X_obs: Incomplete matrix with zeros in missing entries
    :param mask: Boolean array where True indicates observed entries
    :param rank: Number of singular values to keep
    :param lambda_soft: Soft-thresholding value (shrinkage)
    :param max_iters: Max iterations
    :param tol: Convergence threshold on change
    :param verbose: Print progress
    :param constrained: If True, keep observed entries fixed (classic SI); if False, allow full updates (loose SI)
    :return: Completed matrix
    """
    X_filled = X_obs.copy().astype(float)
    prev_X = X_filled.copy()

    for it in range(max_iters):
        # SVD on current filled matrix
        U, S, Vt = np.linalg.svd(X_filled, full_matrices=False)

        # Soft-thresholding on singular values
        S_thresh = np.maximum(S[:rank] - lambda_soft, 0)

        # Reconstruct low-rank matrix
        X_recon = (U[:, :rank] * S_thresh) @ Vt[:rank, :]

        # Either keep observed entries fixed (classic) or allow all entries to update (loose)
        if constrained:
            X_filled[~mask] = X_recon[~mask]  # only update missing entries
        else:
            X_filled = X_recon  # update everything

        # Check for convergence
        diff = np.linalg.norm(X_filled - prev_X, ord='fro')
        if verbose and (it % 10 == 0 or it == 1 or diff < tol):
            print(f"[Soft-Impute {'Constrained' if constrained else 'Loose'}] Iter {it}, Change: {diff:.6f}")
        if diff < tol:
            break

        prev_X = X_filled.copy()

    return X_filled


def compare_solvers(X_obs, X_true, u0, mask, lambda_r, plot=False):
    """
    Compare the performance of Alternating Optimization (AO) and SVD on matrix completion.
    :param X_obs: Incomplete matrix (n x n), where missing entries are zeros
    :param X_true: Ground truth matrix (n x n)
    :param u0: Initial guess for vector u (n,)
    :param mask: Binary mask of observed entries (n x n)
    :param lambda_r: Regularization strength (lambda)
    :param plot: If True, plot the convergence of AO
    """

    # Perform AO
    start = time.time()
    u, v, it, res, hist = alternating_optimization(u0, X_obs, mask, max_it=500, lambda_reg=lambda_r,
                                                   verbose=False,
                                                   track_residuals=plot)
    end = time.time()

    # Distance comparisons
    ao_sol = np.outer(u, v)
    EY = np.linalg.norm((ao_sol - X_obs) * mask, 'fro')
    print(f"AO: Residual={res:.4f}, Distance={EY:.4f}, Iter={it}")

    # Errors
    observed_error_ao = np.linalg.norm((ao_sol - X_true) * mask, ord='fro')
    full_error_ao = np.linalg.norm(ao_sol - X_true, ord='fro')

    print(f"AO error on observed entries: {observed_error_ao:.6f}")
    print(f"AO error on full matrix: {full_error_ao:.6f}")
    print(f"AO time: {end - start:.4f} seconds")

    if hist:
        plt.plot(hist, '--', label='AO Residual')
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.yscale('log')
        plt.title("Convergence of Alternating Optimization")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Perform AO with normalized v
    start = time.time()
    u, v, it, res, hist = alternating_optimization(u0, X_obs, mask, max_it=500, lambda_reg=lambda_r/1000,
                                                   verbose=False,
                                                   track_residuals=plot, norm_v=True)
    end = time.time()

    # Distance comparisons
    ao_sol = np.outer(u, v)
    EY = np.linalg.norm((ao_sol - X_obs) * mask, 'fro')
    print(f"AON: Residual={res:.4f}, Distance={EY:.4f}, Iter={it}")

    # Errors
    observed_error_ao = np.linalg.norm((ao_sol - X_true) * mask, ord='fro')
    full_error_ao = np.linalg.norm(ao_sol - X_true, ord='fro')

    print(f"AON error on observed entries: {observed_error_ao:.6f}")
    print(f"AON error on full matrix: {full_error_ao:.6f}")
    print(f"AON time: {end - start:.4f} seconds")

    if hist:
        plt.plot(hist, '--', label='AO Residual')
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.yscale('log')
        plt.title("Convergence of Alternating Optimization")
        plt.legend()
        plt.grid(True)
        plt.show()


    # SVD solution

    # time the svd
    start = time.time()
    # noinspection PyTypeChecker
    U, S, Vt = svds(csr_matrix(X_obs), k=1)
    sol_svd = (U @ np.diag(S) @ Vt)
    end = time.time()

    observed_error_svd = np.linalg.norm((sol_svd - X_true) * mask, ord='fro')
    full_error_svd = np.linalg.norm(sol_svd - X_true, ord='fro')

    print(f"SVD error on observed entries: {observed_error_svd:.6f}")
    print(f"SVD error on full matrix: {full_error_svd:.6f}")
    print(f"SVD time: {end - start:.4f} seconds")

    # Truncated SVD solution
    # Fill missing entries with mean of observed entries
    X_filled = X_obs.copy()
    start = time.time()
    svd = TruncatedSVD(n_components=1)
    U = svd.fit_transform(X_filled)
    V = svd.components_
    sol_svd = U @ V
    end = time.time()
    observed_error_svd = np.linalg.norm((sol_svd - X_true) * mask, ord='fro')
    full_error_svd = np.linalg.norm(sol_svd - X_true, ord='fro')

    print(f"Truncated SVD error on observed entries: {observed_error_svd:.6f}")
    print(f"Truncated SVD error on full matrix: {full_error_svd:.6f}")
    print(f"Truncated SVD time: {end - start:.4f} seconds")

    # Soft-Impute with constrained and loose options
    # Constrained Soft-Impute, keeping observed entries fixed
    start = time.time()
    X_filled = soft_impute(X_obs, mask, rank=1, lambda_soft=0.01, max_iters=100)
    end = time.time()
    observed_error = np.linalg.norm((X_filled - X_true) * mask, ord='fro') # error on observed entries is always 0
    full_error = np.linalg.norm(X_filled - X_true, ord='fro')
    print(f"Soft-Impute Constr. error on observed entries: {observed_error:.6f}")
    print(f"Soft-Impute Constr. error on full matrix: {full_error:.6f}")
    print(f"Soft-Impute Constr. time: {end - start:.4f} seconds")

    # Loose Soft-Impute, allowing all entries to update
    start = time.time()
    X_filled = soft_impute(X_obs, mask, rank=1, lambda_soft=0.01, max_iters=100, constrained=False)
    end = time.time()
    observed_error = np.linalg.norm((X_filled - X_true) * mask, ord='fro')
    full_error = np.linalg.norm(X_filled - X_true, ord='fro')
    print(f"Soft-Impute Loose error on observed entries: {observed_error:.6f}")
    print(f"Soft-Impute Loose error on full matrix: {full_error:.6f}")
    print(f"Soft-Impute Loose time: {end - start:.4f} seconds")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Parameters
    # Size of the matrix
    n = 100
    # Density of observed entries
    density = 10 / n

    # Ground truth rank-1 matrix
    u_true = np.random.randn(n)
    v_true = np.random.randn(n)
    X_true = np.outer(u_true, v_true)

    # Binary mask of observed entries
    mask = (np.random.rand(n, n) < density)

    # Incomplete observation
    X_obs = X_true * mask

    u0 = np.random.randn(n)  # random init instead of ones

    # Perform lambda search for alternating optimization
    lambda_scan = False
    if lambda_scan:
        # Parameters for lambda sweep
        num_seeds = 20
        lambda_values = np.logspace(-5, 0, 20)

        aggregated_results, results_per_lamda = multi_seed_lambda_sweep(num_seeds, n, density, lambda_values)
        plot_lambda_results(aggregated_results)
        plot_unaggregated_per_seed(results_per_lamda)

    # Regularization parameter
    lambda_r = 4.03e-01

    # Benchmark the different solvers
    compare_solvers(X_obs, X_true, u0.copy(), mask, lambda_r, plot=True)

    # Add noise to observed entries
    noise_std = 0.1
    X_noisy = X_true + noise_std * np.random.randn(*X_true.shape)
    X_noisy_obs = X_noisy * mask

    # Benchmark the different solvers with noise
    print("\n=== Noisy Observations ===")
    compare_solvers(X_noisy_obs, X_true, u0.copy(), mask, lambda_r, plot=True)
