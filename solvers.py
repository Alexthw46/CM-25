import numpy as np


def generate_synthetic_problem(m: int,
                               n: int,
                               density: float,
                               seed: int = None):
    """
    Generate a noisy rank‑1 matrix‑completion problem.

    :param m: number of rows
    :param n: number of columns
    :param density: fraction of entries observed (0 < density <= 1)
    :param seed: optional random seed for reproducibility
    :return:
        X_true:   full rank‑1 matrix (m x n)
        X_obs:    masked observations (zeros where unobserved)
        mask:     boolean mask of observed entries (m x n)
        D:        list of (i, j, X_true[i,j]) for each observed entry
        u_true, v_true: the true generating factors
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample true factors and build true matrix from gaussian distribution
    u_true = np.random.randn(m)
    v_true = np.random.randn(n)
    X_true = np.outer(u_true, v_true)

    # Generate a random observation mask
    mask = (np.random.rand(m, n) < density)

    # Apply the observation mask to the true matrix
    X_obs = X_true * mask

    return X_true, X_obs, mask, u_true, v_true


def initialize_uv(X_obs, mask, strategy='gaussian', epsilon=0, seed=None):
    """
    Initialize u and v based on the specified strategy.

    :param X_obs: Observed matrix (m x n)
    :param mask: Binary mask of observed entries (m x n)
    :param strategy: 'gaussian', 'svd' or 'mean'
    :param epsilon: Noise level for 'svd+noise'
    :param seed: Optional random seed
    :return: (u0, v0)
    """
    if seed is not None:
        np.random.seed(seed)

    m, n = X_obs.shape

    if strategy == 'gaussian':
        u0 = np.random.randn(m)
        v0 = np.random.randn(n)

    elif strategy == 'svd':
        X_filled = X_obs.copy()
        mean_val = X_obs[mask].mean()
        X_filled[~mask] = mean_val

        # SVD initialization
        U, S, VT = np.linalg.svd(X_filled, full_matrices=False)
        # Scale by singular values
        u0 = U[:, 0] * np.sqrt(S[0])
        v0 = VT[0, :] * np.sqrt(S[0])

        if epsilon > 0:
            u0 += epsilon * np.random.randn(m)
            v0 += epsilon * np.random.randn(n)

    elif strategy == 'mean':
        v0 = np.zeros(n)
        for j in range(n):
            obs = mask[:, j]
            v0[j] = X_obs[obs, j].mean() if np.any(obs) else 0.0

        u0 = np.zeros(m)
        for i in range(m):
            obs = mask[i, :]
            v_sub = v0[obs]
            x_row = X_obs[i, obs]
            u0[i] = solve_least_squares(x_row, v_sub) if len(x_row) else 0.0

    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")

    # Normalize v0, rescale u0

    v_norm = np.linalg.norm(v0)

    return u0, v0


def solve_least_squares(x_vals: np.ndarray, y_vals: np.ndarray, lambda_reg: float = 0.0) -> float:
    """
    Solve scalar least-squares: minimize (a * y_vals - x_vals)^2 + lambda_reg * a^2
    Closed-form: a = (x_vals . y_vals) / (y_vals . y_vals + lambda_reg)

    :param x_vals: Observed values (vector)
    :param y_vals: Corresponding factors (vector)
    :param lambda_reg: Regularization parameter
    :return: Optimal scalar a
    """
    # uses dot instead of @ to avoid ambiguity in the type check
    numerator = x_vals.dot(y_vals)
    denominator = y_vals.dot(y_vals) + lambda_reg
    return numerator / denominator


def alternating_optimization(X: np.ndarray, X_mask: np.ndarray, u: np.ndarray, v: np.ndarray = None, max_it: int = 100,
                             eps: float = 1e-12, lambda_reg: float = 1e-8, norm_v: bool = False, verbose: bool = False,
                             track_residuals: bool = False):
    """
    Alternating optimization for rank-1 matrix completion.

    :param X: Data matrix (m x n) with missing entries as zeros
    :param X_mask: Binary mask of observed entries (m x n)
    :param u: Initial guess for vector u (m,).
    :param v: Initial guess for vector v (n,). If None, random init.
    :param max_it: Maximum iterations
    :param eps: Tolerance for improvement
    :param lambda_reg: Regularization strength
    :param norm_v: If True, normalize v (and rescale u) at each iteration
    :param verbose: If True, print progress
    :param track_residuals: If True, track residuals and objective values
    :return:
        u, v: optimized factors
        it: number of iterations
        rec_error: final reconstruction error on observed entries
        histories: dict with keys 'residuals' and 'objective' (if track_residuals else None)
    """
    m, n = X.shape
    if u.shape[0] != m:
        raise ValueError(f"u has length {u.shape[0]}, but X has {m} rows")

    # Initialize v if needed
    if v is None:
        v = np.random.randn(n)
    elif v.shape[0] != n:
        raise ValueError(f"v has length {v.shape[0]}, but X has {n} columns")

    # Optional initial normalization
    if norm_v:
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            v /= v_norm
            u *= v_norm

    it = 0
    prev_res = np.inf
    histories = {'residuals': [], 'objective': []} if track_residuals else None

    while it < max_it:
        it += 1

        # Update v (length n)
        for j in range(n):
            col_idx = X_mask[:, j].astype(bool)
            if not np.any(col_idx):
                v[j] = 0.0
            else:
                x_col = X[col_idx, j]
                u_sub = u[col_idx]
                v[j] = solve_least_squares(x_col, u_sub, lambda_reg)

        # Update u (length m)
        for i in range(m):
            row_idx = X_mask[i, :].astype(bool)
            if not np.any(row_idx):
                u[i] = 0.0
            else:
                x_row = X[i, row_idx]
                v_sub = v[row_idx]
                u[i] = solve_least_squares(x_row, v_sub, lambda_reg)

        # Optional normalization
        if norm_v:
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                v /= v_norm
                u *= v_norm

        # Compute reconstruction error on observed entries
        A_hat = np.outer(u, v)
        diff = (A_hat - X) * X_mask
        rec_error = np.linalg.norm(diff, 'fro')

        # cast to float64 for type consistency
        rec_error = rec_error.astype(np.float64)

        # Full objective value with regularization
        if track_residuals:
            obj_val = rec_error ** 2 + lambda_reg * (np.linalg.norm(u) ** 2 + np.linalg.norm(v) ** 2)
            histories['residuals'].append(rec_error ** 2)
            histories['objective'].append(obj_val)

        improvement = prev_res - rec_error
        if verbose and (it == 1 or it % 10 == 0 or improvement <= eps):
            msg = f"Iter {it:3d}, Residual={rec_error:.6e}"
            if track_residuals:
                msg += f", Obj={obj_val:.6e}"
            msg += f", Improvement={improvement:.2e}"
            print(msg)

        if improvement <= eps:
            break
        prev_res = rec_error

    return u, v, it, rec_error, histories


def gradient_descent_rank1(X, X_mask, u_init=None, v_init=None,
                           max_it=1000, lr=1e-2, lambda_reg=0.0,
                           tol=1e-6, verbose=False, track_residuals=False):
    """
    Gradient descent for rank-1 matrix completion using observed entries mask.

    :param X: Incomplete matrix (m x n), missing entries are zero or ignored
    :param X_mask: Binary mask of observed entries (m x n) Omega, bool
    :param u_init: Optional initialization for u (m,).
    :param v_init: Optional initialization for v (n,).
    :param max_it: Maximum iterations
    :param lr: Learning rate
    :param lambda_reg: Regularization strength (lambda)
    :param tol: Stop if improvement < tol
    :param verbose: Print progress every 10 iterations
    :param track_residuals: Return residual history
    :return: u, v, iters, final_residual, (history if tracked)
    """
    m, n = X.shape
    u = u_init if u_init is not None else np.random.randn(m)
    v = v_init if v_init is not None else np.random.randn(n)

    u = u.astype(np.float64)
    v = v.astype(np.float64)
    X = X.astype(np.float64)
    X_mask = X_mask.astype(bool)

    history = []
    prev_obj = obj = np.inf

    # Precompute indices of observed entries for efficiency
    obs_i, obs_j = np.where(X_mask)

    it = 0

    # Gradient descent loop
    for it in range(1, max_it + 1):
        # Compute prediction only on observed entries
        pred_obs = u[obs_i] * v[obs_j]  # element-wise product u_i * v_j
        resid_obs = pred_obs - X[obs_i, obs_j]  # residual on observed entries

        # Gradients:
        # For u_i: sum over j with (i,j) in Omega of 2 * resid * v_j + 2 * lambda * u_i
        grad_u = np.zeros(m, dtype=np.float64)
        np.add.at(grad_u, obs_i, 2 * resid_obs * v[obs_j])
        grad_u += 2 * lambda_reg * u

        # For v_j: sum over i with (i,j) in Omega of 2 * resid * u_i + 2 * lambda * v_j
        grad_v = np.zeros(n, dtype=np.float64)
        np.add.at(grad_v, obs_j, 2 * resid_obs * u[obs_i])
        grad_v += 2 * lambda_reg * v

        # Gradient descent update
        u -= lr * grad_u
        v -= lr * grad_v

        # Compute objective f_lambda (loss + reg)
        loss = np.sum(resid_obs ** 2)
        reg = lambda_reg * (np.dot(u, u) + np.dot(v, v))
        obj = loss + reg

        if track_residuals:
            history.append(obj)

        if verbose and (it % 10 == 0 or it == 1):
            print(f"[GD] Iter {it}, Objective: {obj:.6f}")

        # Stopping criterion: check improvement
        if np.abs(prev_obj - obj) < tol:
            break
        prev_obj = obj

    if track_residuals:
        return u, v, it, obj, history
    else:
        return u, v, it, obj, None