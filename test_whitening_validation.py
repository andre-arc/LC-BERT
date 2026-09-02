"""
Validate the whitening kernels against the defining constraint of whitening.

This test exercises the SAME code path the training pipeline uses. The pipeline
applies kernels to ROW vectors:

    z = (x + bias) . kernel          # data_utils/ag_news/whitening.py

so for the transform to be a whitening transform the kernel W must satisfy

    W^T . Sigma . W = I

(equivalently W^T W = Sigma^-1 for the column-vector form W^T used by
Kessy et al. 2018, Eq. 3). An earlier version of this script applied
`pca_matrix.T` instead of `pca_matrix`, so it validated the transpose of what
the pipeline actually computes and could not detect an orientation error.

Run:  python test_whitening_validation.py
"""

import numpy as np

from data_utils.ag_news.whitening import BertWhiteningDataset


class KernelMath:
    """
    Borrow the kernel math from BertWhiteningDataset without constructing a
    dataset, so no tokenizer, model or GPU is required.
    """
    epsilon = BertWhiteningDataset.EPSILON

    _covariance = BertWhiteningDataset._covariance
    _fix_signs = BertWhiteningDataset._fix_signs
    _sorted_eigh = BertWhiteningDataset._sorted_eigh
    _normalize = BertWhiteningDataset._normalize
    _transform_and_normalize = BertWhiteningDataset._transform_and_normalize

    _compute_kernel_bias_svd = BertWhiteningDataset._compute_kernel_bias_svd
    _compute_kernel_bias_eigen = BertWhiteningDataset._compute_kernel_bias_eigen
    _compute_kernel_bias_pca = BertWhiteningDataset._compute_kernel_bias_pca
    _compute_kernel_bias_pca_svd = BertWhiteningDataset._compute_kernel_bias_pca_svd
    _compute_kernel_bias_pca_cor = BertWhiteningDataset._compute_kernel_bias_pca_cor
    _compute_kernel_bias_zca_base = BertWhiteningDataset._compute_kernel_bias_zca_base
    _compute_kernel_bias_zca_svd = BertWhiteningDataset._compute_kernel_bias_zca_svd


TECHNIQUES = {
    'svd (BERT-Whitening)': '_compute_kernel_bias_svd',
    'pca': '_compute_kernel_bias_pca',
    'pca-cor': '_compute_kernel_bias_pca_cor',
    'pca-svd': '_compute_kernel_bias_pca_svd',
    'zca': '_compute_kernel_bias_zca_base',
    'zca-svd': '_compute_kernel_bias_zca_svd',
    'eigen': '_compute_kernel_bias_eigen',
}

N_FEATURES = 10
TARGET_DIM = 6          # stands in for the pipeline's 768 -> 256 truncation
TOLERANCE = 1e-5


def make_correlated_data(n_samples=5000, n_features=N_FEATURES, seed=42):
    """Correlated Gaussian data with a positive-definite covariance."""
    rng = np.random.RandomState(seed)
    a = rng.rand(n_features, n_features)
    cov = a @ a.T + np.eye(n_features) * 0.5
    return rng.multivariate_normal(np.zeros(n_features), cov, n_samples)


def whitened_covariance(math, X, kernel, bias):
    """Covariance of exactly what the pipeline computes, minus the L2 step."""
    Z = (X + bias).dot(kernel)
    return math._covariance(Z)


def report(name, cov, k):
    """Print and grade a whitened covariance against the identity."""
    deviation = np.abs(cov - np.eye(k)).max()
    ok = deviation < TOLERANCE
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} "
          f"max|Cov(z) - I| = {deviation:.2e}")
    return ok


def main():
    math = KernelMath()
    X = make_correlated_data()
    sigma = math._covariance(X)
    failures = []

    print("=" * 78)
    print("WHITENING VALIDATION - pipeline convention  z = (x + bias) . kernel")
    print("=" * 78)
    print(f"\nData: {X.shape[0]} samples x {X.shape[1]} features, "
          f"epsilon = {math.epsilon:g}")
    print(f"Condition number of Sigma: {np.linalg.cond(sigma):.1f}")

    print("\n1. Full-rank kernels must satisfy W^T Sigma W = I")
    print("-" * 78)
    for label, method in TECHNIQUES.items():
        kernel, bias = getattr(math, method)([X])
        assert kernel.ndim == 2, f"{label}: kernel collapsed to shape {kernel.shape}"
        cov = whitened_covariance(math, X, kernel, bias)
        if not report(label, cov, N_FEATURES):
            failures.append(label)

    print(f"\n2. Truncated to {TARGET_DIM} of {N_FEATURES} components, must give I_k")
    print("-" * 78)
    for label, method in TECHNIQUES.items():
        kernel, bias = getattr(math, method)([X])
        kernel = kernel[:, :TARGET_DIM]
        cov = whitened_covariance(math, X, kernel, bias)
        if not report(label, cov, TARGET_DIM):
            failures.append(f"{label} (truncated)")

    print("\n3. Compression ordering - Kessy et al. Propositions 3 and 4")
    print("-" * 78)
    print("  The per-component integration measure must be non-increasing, or")
    print("  truncation does not keep the leading components. Proposition 3 scores")
    print("  PCA-type kernels on cross-covariance, Proposition 4 scores the")
    print("  scale-invariant cor variants on cross-correlation.")
    sd_x = np.sqrt(np.diag(sigma))
    for label, scale_invariant in (('svd (BERT-Whitening)', False),
                                   ('pca', False),
                                   ('pca-cor', True)):
        kernel, bias = getattr(math, TECHNIQUES[label])([X])
        cross = kernel.T.dot(sigma)                     # F = W^T Sigma  (Eq. 6)
        if scale_invariant:
            cross = cross / sd_x                        # Psi = F V^-1/2 (Eq. 7)
        measure = (cross ** 2).sum(axis=1)
        monotone = np.all(np.diff(measure) <= 1e-8)
        symbol = 'psi' if scale_invariant else 'phi'
        print(f"  {'PASS' if monotone else 'FAIL'}  {label:<24} "
              f"{symbol} = {np.array2string(measure[:4], precision=2)} ...")
        if not monotone:
            failures.append(f"{label} (ordering)")

    print("\n4. Basis stability across splits - Kessy et al. sec. 2 and 5")
    print("-" * 78)
    print("  Eigenvectors are defined only up to a sign, so a kernel fitted on one")
    print("  split must not be an arbitrary sign flip of one fitted on another.")
    print("  Compares the leading 3 columns, where eigenvalues are well separated")
    print("  and the component order is stable between splits.")
    half = len(X) // 2
    for label in ('pca', 'zca', 'svd (BERT-Whitening)'):
        k_a, _ = getattr(math, TECHNIQUES[label])([X[:half]])
        k_b, _ = getattr(math, TECHNIQUES[label])([X[half:]])
        k_a, k_b = k_a[:, :3], k_b[:, :3]
        aligned = np.abs(k_a - k_b).max()
        flipped = np.abs(k_a + k_b).max()
        stable = aligned < flipped
        print(f"  {'PASS' if stable else 'FAIL'}  {label:<24} "
              f"max|Wa - Wb| = {aligned:.2e} vs max|Wa + Wb| = {flipped:.2e}")
        if not stable:
            failures.append(f"{label} (sign stability)")

    print("\n5. Regression check - the pre-fix PCA orientation")
    print("-" * 78)
    print("  Lambda^-1/2 V^T is correct for the COLUMN convention z = W x, but the")
    print("  pipeline uses row vectors. Applied there it must fail the constraint;")
    print("  if this passes, the orientation fix has been reverted.")
    w, v = math._sorted_eigh(sigma)
    w = np.maximum(w, math.epsilon)
    diagw = np.diag(1.0 / np.sqrt(w + math.epsilon))
    transposed_kernel = diagw.dot(v.T)                  # the old _compute_kernel_bias_pca
    mu = X.mean(axis=0, keepdims=True)
    cov_bad = whitened_covariance(math, X, transposed_kernel, -mu)
    deviation = np.abs(cov_bad - np.eye(N_FEATURES)).max()
    if deviation > TOLERANCE:
        print(f"  PASS  transposed PCA kernel     max|Cov(z) - I| = {deviation:.2e} "
              f"(correctly rejected)")
    else:
        print(f"  FAIL  transposed PCA kernel is being accepted as whitening")
        failures.append('regression: transposed PCA accepted')

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    else:
        print("All whitening kernels satisfy W^T Sigma W = I in the pipeline's own")
        print("convention, are ordered for compression, and are sign-stable.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
