# Epsilon Control in Whitening Transformations

## Overview

The epsilon parameter has been centralized in the `BertWhiteningDataset` class for easy control of numerical stability across all whitening transformations.

## Default Value

```python
BertWhiteningDataset.EPSILON = 1e-5  # Class-level default
```

## Usage

### Option 1: Use Default Epsilon (1e-5)

```python
# No need to specify epsilon parameter
dataset = BertWhiteningDataset(
    device=device,
    dataset=data,
    tokenizer=tokenizer,
    model=model,
    max_len=128,
    dim_technique='zca'
)
```

### Option 2: Specify Custom Epsilon

```python
# Provide custom epsilon value
dataset = BertWhiteningDataset(
    device=device,
    dataset=data,
    tokenizer=tokenizer,
    model=model,
    max_len=128,
    dim_technique='zca',
    epsilon=1e-6  # Custom value for more/less stability
)
```

### Option 3: Change Class Default Globally

```python
# Change default for all future instances
BertWhiteningDataset.EPSILON = 1e-6

# All new instances will use 1e-6 by default
dataset = BertWhiteningDataset(...)
```

## Where Epsilon is Used

Epsilon is added for numerical stability in the following whitening methods:

1. **PCA (Eigendecomposition)**
   - Line 156: `diagw = np.diag(1/((w + self.epsilon)**0.5))`
   - Prevents division by zero when eigenvalues are very small

2. **PCA-SVD**
   - Line 179: `diag_sigma_inv = np.diag(1 / (diag_sigma**0.5 + self.epsilon))`
   - Stabilizes singular value inversion

3. **ZCA (with sklearn PCA)**
   - Line 206: `diagw = np.diag(1/((w + self.epsilon)**0.5))`
   - Prevents numerical instability in eigenvalue inversion

4. **ZCA Base (Eigendecomposition)**
   - Line 235: `diagw = np.diag(1/((w + self.epsilon)**0.5))`
   - Adds stability to ZCA whitening matrix computation

5. **ZCA-SVD**
   - Line 259: `diag_sigma_inv = np.diag(1 / (diag_sigma**0.5 + self.epsilon))`
   - Prevents numerical issues in SVD-based ZCA

## Recommended Values

- **Default (1e-5)**: Good balance for most use cases
- **More stability (1e-4 or 1e-3)**: Use if you encounter numerical issues
- **Less regularization (1e-6 or 1e-7)**: Use if whitening is too conservative

## Impact on Results

- **Larger epsilon**: More regularization, less aggressive whitening, more stable
- **Smaller epsilon**: Less regularization, more aggressive whitening, may be numerically unstable

## Affected Whitening Techniques

The following techniques use epsilon:
- `pca` - PCA whitening via eigendecomposition
- `pca-svd` - PCA whitening via SVD
- `zca` - ZCA whitening with sklearn PCA
- `zca` (base) - ZCA whitening via eigendecomposition
- `zca-svd` - ZCA whitening via SVD

The following techniques do NOT use epsilon (stable without it):
- `svd` - Basic SVD whitening
- `eigen` - Basic eigendecomposition whitening

## Example: Testing Different Epsilon Values

```python
# Test different epsilon values
epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

for eps in epsilons:
    print(f"\nTesting epsilon = {eps}")

    dataset = BertWhiteningDataset(
        device=device,
        dataset=train_data,
        tokenizer=tokenizer,
        model=model,
        max_len=128,
        dim_technique='zca',
        epsilon=eps
    )

    # Train and evaluate model...
    # Compare F1 scores
```
