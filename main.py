import time
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import plot_lambda_results, plot_unaggregated_per_seed, plot_3d_surface, top_k_results


def alternating_optimization(u, X, X_mask, max_it=100, eps=1e-12, lambda_reg=1e-8, v=None, verbose=False,
                             track_residuals=False, norm_v=False):
    """
    Alternating optimization for matrix completion.

    :param u: Initial guess for vector u (n,)
    :param v: Initial guess for vector v (n,)
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
    if v is None:
        v = np.random.randn(n)

    # Normalize v and adjust u
    v_norm = np.linalg.norm(v)
    v /= v_norm
    u *= v_norm

    it = 0
    prev_res = 1e12
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


def gradient_descent_rank1(X, X_mask, u_init=None, v_init=None,
                           max_it=1000, lr=1e-2, lambda_reg=0.0,
                           tol=1e-6, verbose=False, track_residuals=False):
    """
    Gradient descent for rank-1 matrix completion.

    :param X: Incomplete matrix (n x n), missing entries are zero
    :param X_mask: Binary mask of observed entries (n x n)
    :param u_init: Optional initialization for u (n,)
    :param v_init: Optional initialization for v (n,)
    :param max_it: Maximum iterations
    :param lr: Learning rate
    :param lambda_reg: Regularization strength (lambda)
    :param tol: Stop if improvement < tol
    :param verbose: Print progress
    :param track_residuals: Return residual history
    :return: u, v, iters, residual, (residual_history if tracked)
    """

    n = X.shape[0]
    u = u_init if u_init is not None else np.random.randn(n)
    v = v_init if v_init is not None else np.random.randn(n)

    residual_history = []
    prev_residual = np.inf

    for it in range(1, max_it + 1):
        A_hat = np.outer(u, v)
        residual = (A_hat - X) * X_mask

        # Calcola i gradienti in float64 per evitare overflow numerici
        grad_u = 2 * (residual.astype(np.float64) @ v.astype(np.float64)) + 2 * lambda_reg * u.astype(np.float64)
        grad_v = 2 * (residual.T.astype(np.float64) @ u.astype(np.float64)) + 2 * lambda_reg * v.astype(np.float64)

        # Update step
        u -= lr * grad_u
        v -= lr * grad_v

        # Compute loss
        res_norm = np.linalg.norm(residual, 'fro')

        if track_residuals:
            residual_history.append(res_norm)

        if verbose and (it % 10 == 0 or it == 1):
            print(f"[GD] Iter {it}, Residual: {res_norm:.6f}")

        if np.abs(prev_residual - res_norm) < tol:
            break
        prev_residual = res_norm

    return u, v, it, res_norm, residual_history if track_residuals else None


def gradient_grid_sweep(
    num_seeds, n, density,
    lambda_values, lr_values,
    grad_solver,
    max_it=1000, eps=1e-6
):
    """
    Evaluate gradient descent solver over a grid of (lambda, learning rate) pairs.

    :param num_seeds: Number of random seeds to average over
    :param n: Matrix size
    :param density: Observation density
    :param lambda_values: List of regularization values to scan
    :param lr_values: List of learning rates to scan
    :param grad_solver: Gradient-based solver function with signature:
        (u0, v0, X, X_mask, lr, lambda_reg, max_it, eps)
    :param max_it: Max iterations
    :param eps: Convergence threshold
    :return: Dictionary with metrics for each (lambda, lr) pair
    """

    results = {}

    for seed in range(num_seeds):
        np.random.seed(seed)

        print(f"\n=== Seed {seed} ===")

        # Ground truth and observed data
        u_true = np.random.randn(n)
        v_true = np.random.randn(n)
        X_true = np.outer(u_true, v_true)
        mask = (np.random.rand(n, n) < density)
        X_obs = X_true * mask

        for lam in lambda_values:
            for lr in lr_values:
                key = (lam, lr)
                if key not in results:
                    results[key] = {'obs_errors': [], 'full_errors': [], 'iters': []}

                u0 = np.random.randn(n)
                v0 = np.random.randn(n)

                u, v, iters, _, _ = grad_solver(
                    X_obs, mask,
                    u_init=u0.copy(), v_init=v0.copy(),
                    lr=lr,
                    lambda_reg=lam,
                    max_it=max_it,
                    tol=eps
                )

                A_hat = np.outer(u, v)
                obs_error = np.linalg.norm((A_hat - X_true) * mask, ord='fro')
                full_error = np.linalg.norm(A_hat - X_true, ord='fro')

                results[key]['obs_errors'].append(obs_error)
                results[key]['full_errors'].append(full_error)
                results[key]['iters'].append(iters)

    # Aggregate stats
    aggregated = []
    for (lam, lr), res in results.items():
        aggregated.append({
            'lambda': lam,
            'lr': lr,
            'obs_mean': np.mean(res['obs_errors']),
            'obs_std': np.std(res['obs_errors']),
            'full_mean': np.mean(res['full_errors']),
            'full_std': np.std(res['full_errors']),
            'iters_mean': np.mean(res['iters']),
            'iters_std': np.std(res['iters']),
        })

    return aggregated, results


def als_lambda_sweep(num_seeds, n, density, lambda_values, maxit=500, eps=1e-6, norm_v=True):
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
        v0 = np.random.randn(n)

        print(f"\n=== Seed {seed} ===")
        for lam in lambda_values:
            u, v, it, res, _ = alternating_optimization(
                u0.copy(), X_obs, mask, v=v0.copy(), max_it=maxit, eps=eps, lambda_reg=lam, verbose=False,
                track_residuals=False,
                norm_v=norm_v
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


def compare_solvers(X_obs, X_true, u0, v0, mask, lambda_r, plot=False):
    """
    Compare the performance of Alternating Optimization (AO) and SVD on matrix completion.
    :param X_obs: Incomplete matrix (n x n), where missing entries are zeros
    :param X_true: Ground truth matrix (n x n)
    :param u0: Initial guess for vector u (n,)
    :param v0: Initial guess for vector v (n,)
    :param mask: Binary mask of observed entries (n x n)
    :param lambda_r: Regularization strength (lambda)
    :param plot: If True, plot the convergence of AO
    """

    # Perform AO
    start = time.time()
    u, v, it, res, hist = alternating_optimization(u0.copy(), X_obs, mask, max_it=200, lambda_reg=lambda_r,
                                                   v=v0.copy(), verbose=False,
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

    # Perform AO with norm constraint
    start = time.time()
    u, v, it, res, hist2 = alternating_optimization(u0.copy(), X_obs, mask, max_it=200, lambda_reg=1e-02,
                                                    v=v0.copy(), verbose=False,
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

    if hist2 and hist:
        plt.plot(hist, '--', label='AO Residual')
        plt.plot(hist2, '--', label='AON Residual')
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.yscale('log')
        plt.title("Convergence of Alternating Optimization")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Perform Gradient Descent
    start = time.time()

    lr = 9.41e-04
    lambda_reg = 6.95e-05
    u, v, it, res, hist_3 = gradient_descent_rank1(X_obs, mask, u_init=u0.copy(), v_init=v0.copy(),
                                                   max_it=1500, lr=lr, lambda_reg=lambda_reg,
                                                   tol=1e-8, verbose=False, track_residuals=plot)
    end = time.time()

    # Distance comparisons
    gd_sol = np.outer(u, v)
    EY = np.linalg.norm((gd_sol - X_obs) * mask, 'fro')
    print(f"GD: Residual={res:.4f}, Distance={EY:.4f}, Iter={it}")
    # Errors
    observed_error_gd = np.linalg.norm((gd_sol - X_true) * mask, ord='fro')
    full_error_gd = np.linalg.norm(gd_sol - X_true, ord='fro')
    print(f"GD error on observed entries: {observed_error_gd:.6f}")
    print(f"GD error on full matrix: {full_error_gd:.6f}")
    print(f"GD time: {end - start:.4f} seconds")

    if hist_3:
        plt.plot(hist_3, '--', label='GD Residual')
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.yscale('log')
        plt.title("Convergence of Gradient Descent")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Truncated SVD solution
    # Fill missing entries with mean of observed entries
    X_filled = X_obs.copy()
    X_filled[~mask] = X_obs[mask].mean()
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

def testGD():
    # Perform grid search for gradient descent
    num_seeds = 40
    lambda_values = np.logspace(-5, -1, 20)
    lr_values = np.logspace(-3.75, -2.5, 20)
    aggregated_results, results_per_lamda = gradient_grid_sweep(
        num_seeds, n, density,
        lambda_values, lr_values,
        grad_solver=gradient_descent_rank1,
        max_it=1000, eps=1e-6
    )
    plot_3d_surface(aggregated_results)

    best_results = top_k_results(aggregated_results, k=5, sort_by='full_mean')

    for i, res in enumerate(best_results, 1):
        print(
            f"#{i}: lambda={res['lambda']:.2e}, lr={res['lr']:.2e}, full_err={res['full_mean']:.4f}, obs_err={res['obs_mean']:.4f}, iters={res['iters_mean']:.1f}")

if __name__ == '__main__':
    # Parameters
    # Size of the matrix
    n = 100
    # Density of observed entries
    density = 10 / n

    # Perform lambda search for alternating optimization
    lambda_scan = False
    if lambda_scan:
        # Parameters for lambda sweep
        num_seeds = 100
        lambda_values = np.logspace(-5, -1, 20)

        aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, density, lambda_values,
                                                                 norm_v=False)
        plot_lambda_results(aggregated_results)
        plot_unaggregated_per_seed(results_per_lamda)

        # aggregated_results, results_per_lamda = multi_seed_lambda_sweep(num_seeds, n, density, lambda_values, false)
        # plot_lambda_results(aggregated_results)
        # plot_unaggregated_per_seed(results_per_lamda)

        exit(0)

    if False:
        testGD()
        exit(0)

    # seed
    seed = int(time.time())
    np.random.seed(seed)

    # Ground truth rank-1 matrix
    u_true = np.random.randn(n)
    v_true = np.random.randn(n)
    X_true = np.outer(u_true, v_true)

    # Binary mask of observed entries
    mask = (np.random.rand(n, n) < density)

    # Incomplete observation
    X_obs = X_true * mask

    u0 = np.random.randn(n)  # random init instead of ones
    v0 = np.random.randn(n)

    # Regularization parameter
    lambda_r = 0.403

    # Benchmark the different solvers
    compare_solvers(X_obs, X_true, u0.copy(), v0.copy(), mask, lambda_r, plot=True)

    # Add noise to observed entries
    noise_std = 0.1
    X_noisy = X_true + noise_std * np.random.randn(*X_true.shape)
    X_noisy_obs = X_noisy * mask

    # Benchmark the different solvers with noise
    print("\n=== Noisy Observations ===")
    compare_solvers(X_noisy_obs, X_true, u0.copy(), v0.copy(), mask, lambda_r, plot=True)

