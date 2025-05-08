# 1. Introduction

## 1.1 Observed Data

Let M ∈ ℝ^{m × n} be an unknown matrix. Instead of the full matrix, we are given a subset of entries:

 
 
𝒟 = { (iₖ, jₖ, xₖ) ∈ {1,…,m} × {1,…,n} × ℝ }ₖ=1,…,p where xₖ = M_{iₖ jₖ}.
Our goal is to approximate M with a rank-1 matrix using this incomplete information.

## 1.2 Rank-1 Matrix Approximation

We seek vectors u ∈ ℝ^m and v ∈ ℝ^n such that A = u vᵀ
minimizes the squared error:
min_{u,v} ∑ₖ (u_{iₖ} v_{jₖ} - xₖ)²

This reduces to the classic Eckart–Young–Mirsky theorem if all entries are available.

## 1.3 Matrix Representation and Sampling Operator

We define a sampling operator 𝒫_Ω that extracts the observed entries:
𝒫_Ω(A) = { A_{iₖ jₖ} }ₖ=1,…,p

Let Ω = { (iₖ, jₖ) } be the index set of observations.

The objective becomes: min_{u,v} ||𝒫_Ω(u vᵀ) - x||²₂

## 1.4 Scale Ambiguity and Normalization

The decomposition A = u vᵀ is not unique due to scaling: for any nonzero α,
 
(αu)(vᵀ / α) = u vᵀ

To eliminate this ambiguity, we may fix a normalization such as ||v|| = 1. This is always achievable by rescaling u.

Note: constraints like ||u||² + ||v||² = 1 are not always feasible (see the m = n = 1, u = 100, v = 1 case).

# 2. Mathematical Analysis

## 2.1 Objective Structure: Bi-convexity
The objective function f(u, v) = ∑ₖ (u_{iₖ} v_{jₖ} - xₖ)²
is not jointly convex, but bi-convex:

* f(·, v) is convex in u for fixed v

* f(u, ·) is convex in v for fixed u

This allows optimization using Alternating Least Squares (ALS).

## 2.2 Gradient Computation
Partial derivatives are:

* ∂f/∂uᵢ = 2 ∑_{k : iₖ = i} (uᵢ v_{jₖ} - xₖ)v_{jₖ}
* ∂f/∂vⱼ = 2 ∑_{k : jₖ = j} (u_{iₖ} vⱼ - xₖ)u_{iₖ}

Efficient to compute using only observed entries.

## 2.3 Hessian and Non-Convexity
The full Hessian of f(u, v) is not positive semidefinite, due to bilinear terms.

This implies the landscape may contain local minima and saddle points — global convergence is not guaranteed.

## 2.4 Solution Uniqueness, Normalization, and Regularization
Scale ambiguity is broken by imposing valid normalizations, such as ||v|| = 1. Improper constraints (e.g. ||u||² + ||v||² = 1) may not be satisfiable via rescaling.

If observations are sparse, multiple solutions can match the data.

As an alternative, we use ℓ₂ regularization to stabilize optimization:
f_λ(u, v) = f(u, v) + λ (||u||² + ||v||²)

This penalizes large magnitudes but doesn't enforce a specific norm.

Note: Regularization improves generalization but does not eliminate scale ambiguity alone.

## 2.5 Degrees of Freedom
We estimate m + n parameters from p observations. To avoid underdetermined systems:
 
p ≫ m + n
must hold for identifiability and stability.

# References
Eckart, C., & Young, G. (1936). "The approximation of one matrix by another of lower rank."

Candès, E. J., & Recht, B. (2009). "Exact matrix completion via convex optimization."
 
