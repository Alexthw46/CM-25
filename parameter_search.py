import numpy as np

from solvers import alternating_optimization, gradient_descent_rank1, generate_synthetic_problem, initialize_uv
from plot_utils import plot_3d_surface, top_k_results, plot_lambda_results, plot_unaggregated_per_seed


def testGD():
    # Perform grid search for gradient descent
    num_seeds = 50
    lambda_values = np.logspace(-1.25, 0, 20)
    lr_values = np.logspace(-2.75, -2.0, 20)

    # Gaussian initialization
    aggregated_results, results_per_lamda = gradient_grid_sweep(
        num_seeds,
        lambda_values, lr_values,
        grad_solver=gradient_descent_rank1,
        max_it=1000, eps=1e-6
    )
    plot_3d_surface(aggregated_results)

    best_results = top_k_results(aggregated_results, k=5, sort_by='full_mean')

    print("\n=== Top 5 results Gaussian ===")
    for i, res in enumerate(best_results, 1):
        print(
            f"#{i}: lambda={res['lambda']:.2e}, lr={res['lr']:.2e}, full_err={res['full_mean']:.4f}, obs_err={res['obs_mean']:.4f}, iters={res['iters_mean']:.1f}")

    # SVD initialization
    lambda_values = np.logspace(-8, -1.25, 20)
    lr_values = np.logspace(-2.6, -1.75, 20)

    aggregated_results, results_per_lamda = gradient_grid_sweep(
        num_seeds,
        lambda_values, lr_values,
        grad_solver=gradient_descent_rank1,
        max_it=1000, eps=1e-6,
        init='svd'
    )
    plot_3d_surface(aggregated_results)
    best_results = top_k_results(aggregated_results, k=5, sort_by='full_mean')
    print("\n=== Top 5 results SVD ===")
    for i, res in enumerate(best_results, 1):
        print(
            f"#{i}: lambda={res['lambda']:.2e}, lr={res['lr']:.2e}, full_err={res['full_mean']:.4f}, obs_err={res['obs_mean']:.4f}, iters={res['iters_mean']:.1f}")

    # SVD initialization with noise
    aggregated_results, results_per_lamda = gradient_grid_sweep(
        num_seeds,
        lambda_values, lr_values,
        grad_solver=gradient_descent_rank1,
        max_it=1000, eps=1e-6,
        init='svd', noise=0.1
    )
    plot_3d_surface(aggregated_results)
    best_results = top_k_results(aggregated_results, k=5, sort_by='full_mean')
    print("\n=== Top 5 results SVD with noise ===")
    for i, res in enumerate(best_results, 1):
        print(
            f"#{i}: lambda={res['lambda']:.2e}, lr={res['lr']:.2e}, full_err={res['full_mean']:.4f}, obs_err={res['obs_mean']:.4f}, iters={res['iters_mean']:.1f}")

    # Mean initialization
    lambda_values = np.logspace(-1.5, -0.25, 20)
    lr_values = np.logspace(-5., -2.4, 20)

    aggregated_results, results_per_lamda = gradient_grid_sweep(
        num_seeds,
        lambda_values, lr_values,
        grad_solver=gradient_descent_rank1,
        max_it=1000, eps=1e-6,
        init='mean'
    )
    plot_3d_surface(aggregated_results)
    best_results = top_k_results(aggregated_results, k=5, sort_by='full_mean')
    print("\n=== Top 5 results Mean ===")
    for i, res in enumerate(best_results, 1):
        print(
            f"#{i}: lambda={res['lambda']:.2e}, lr={res['lr']:.2e}, full_err={res['full_mean']:.4f}, obs_err={res['obs_mean']:.4f}, iters={res['iters_mean']:.1f}")

def gradient_grid_sweep(
        num_seeds,
        lambda_values, lr_values,
        grad_solver,
        max_it=1000, eps=1e-6,
        init='gaussian', noise=0.0
):
    """
    Evaluate gradient descent solver over a grid of (lambda, learning rate) pairs.

    :param init: Initialization strategy for u and v
    :param noise: Noise level to add to the svd initialization
    :param num_seeds: Number of random seeds to average over
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
        X_true, X_obs, mask, u_true, v_true = generate_synthetic_problem(n, m, density,
                                                                         seed + 1)
        # seed+1 to avoid overlap gaussian init with the true uv seed

        u0, v0 = initialize_uv(X_obs, mask, strategy=init,
                               seed=seed, epsilon=noise)

        print(f"\n=== Seed {seed} ===")

        for lam in lambda_values:
            for lr in lr_values:
                key = (lam, lr)
                if key not in results:
                    results[key] = {'obs_errors': [], 'full_errors': [], 'iters': []}

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


def als_lambda_sweep(num_seeds, n, m, density, lambda_values, maxit=500, eps=1e-6, norm_v=True, init='gaussian',
                     noise=0.0):
    """
    Perform tests on a set of lambda values across multiple seeds to assess average performance.

    :param noise:
    :param init: Initialization strategy for u and v
    :param num_seeds: Number of different random seeds to test
    :param n: Number of rows in the matrix
    :param m: Number of columns in the matrix
    :param density: Probability of observing each entry
    :param lambda_values: List of lambda values to test
    :param maxit: Max iterations for alternating optimization
    :param eps: Convergence tolerance
    :return: dict mapping lambda to aggregated results (mean/std of errors and iterations)
    """
    results_per_lambda = {lam: {'obs_errors': [], 'full_errors': [], 'iters': []} for lam in lambda_values}

    for seed in range(num_seeds):

        X_true, X_obs, mask, u_true, v_true = generate_synthetic_problem(n, m, density,
                                                                         seed + 1)
        # seed+1 to avoid overlap gaussian init with the true uv seed

        u0, v0 = initialize_uv(X_obs, mask, strategy=init,
                               seed=seed, epsilon=noise)
        print(f"\n=== Seed {seed} ===")
        for lam in lambda_values:
            u, v, it, res, _ = alternating_optimization(X_obs, mask, u0.copy(), v=v0.copy(), max_it=maxit, eps=eps,
                                                        lambda_reg=lam, norm_v=norm_v, verbose=False,
                                                        track_residuals=False)

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


def test_ALS():
    # Parameters for lambda sweep
    num_seeds = 50
    print(" === Gaussian Initialization ===")
    lambda_values = np.logspace(-3, 0, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values, norm_v=False)
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== === Norm applied === ===")
    lambda_values = np.logspace(-5, -1.5, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values, norm_v=True)
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== SVD Initialization without Noise ===")
    lambda_values = np.logspace(-8, 0, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values, norm_v=False,
                                                             init='svd')
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== === Norm applied === ===")
    lambda_values = np.logspace(-8, -1.5, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values, norm_v=True,
                                                             init='svd')
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== SVD Initialization with Noise ===")
    lambda_values = np.logspace(-12, -5, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values,
                                                             norm_v=False, init='svd', noise=0.1)
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== === Norm applied === ===")
    lambda_values = np.logspace(-12, -5, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values,
                                                             norm_v=True, init='svd', noise=0.1)
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== Mean Initialization ===")
    lambda_values = np.logspace(-3, 0, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values, norm_v=False,
                                                             init='mean')
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)
    print("\n=== === Norm applied === ===")
    lambda_values = np.logspace(-5, -1.5, 20)
    aggregated_results, results_per_lamda = als_lambda_sweep(num_seeds, n, m, density, lambda_values, norm_v=True,
                                                             init='mean')
    plot_lambda_results(aggregated_results)
    plot_unaggregated_per_seed(results_per_lamda)


if __name__ == "__main__":
    # Parameters
    # Size of the matrix
    n = 100
    m = 100
    # Density of observed entries
    density = 10 / n

    # Perform lambda search for alternating optimization
    switch = False #True for ALS, False for GD
    if switch:
        test_ALS()
    else:
        testGD()
