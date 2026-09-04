import sys, os

# Get the directory two levels up from the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to path and change directory
sys.path.append(parent_dir)
os.chdir(parent_dir)

import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import clean
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


class BertWhiteningDataset(Dataset):
    LABEL2INDEX = {'world': 0, 'sports': 1, 'business': 2, 'science': 3}
    INDEX2LABEL = {0: 'world', 1: 'sports', 2: 'business', 3: 'science'}
    NUM_LABELS = 4
    EPSILON = 1e-8  # Epsilon for numerical stability in whitening transformations

    def __init__(self, device, dataset, tokenizer, model, max_len, dim_technique, epsilon=None,
                 whitening_params=None, *args, **kwargs):
        self.device = device
        self.data = self.load_dataset(dataset)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        # Use provided epsilon or fall back to class default
        self.epsilon = epsilon if epsilon is not None else self.EPSILON
        # (kernel, bias) fitted on this split, exposed so other splits can reuse it
        self.whitening_params = None
        self.vecs = self.Dim_reduction(dim_technique=dim_technique, max_len=max_len,
                                       whitening_params=whitening_params)
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        vecs = torch.from_numpy(self.vecs[index]).unsqueeze(0).float()
        vecs, seq_label = vecs, data['target']
        
        return  vecs, np.array(seq_label), data['cleaned_text']
    
    def __len__(self):
        return len(self.data)

    def load_dataset(self, dataset, apply_cleaning=False):
        # dataset = []
        # Read file
        # data = pd.read_csv(path)
        data = dataset

        # label encoder
        class_names = set(data.iloc[:,0].values)
        le = LabelEncoder()
        le.fit(data.iloc[:,0])
        class_names = le.classes_

        data['target'] = le.transform(data.iloc[:,0])

        if apply_cleaning:
            data['cleaned_text'] = clean(data['Description'])
        else:
            data['cleaned_text'] = data['Description']

        # return data.iloc[:1000,:]
        return data
    
    # def get_embeddings(self, max_len):
        
    #     torch.cuda.empty_cache()
        
    #     if(self.tokenizer is None or self.model is None):
    #         raise Exception("Sorry, But You must define Tokenizer and Model")
    
    #     vecs = []
    #     with torch.no_grad():
    #       i = 1
    #       for line in self.data['cleaned_text']:
    #         sentence = line
    #         inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
    #         inputs['input_ids'] = inputs['input_ids'].to(self.device)
    #         inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

    #         hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)[0]
    #         embedding = hidden_states[:, 0, :]
    #         vec = embedding.cpu().numpy()
    
    #         vecs.append(vec)
    #         print(f'\rWord Embedding Process: {i} / {len(self.data["cleaned_text"])} sentences | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
    #         i+=1
    
    #     return np.array(vecs)

    def _transform_and_normalize(self, vecs, kernel, bias):
        """
            Applying transformation then standardize
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
            # vecs = (vecs + bias) @ kernel
        return self._normalize(vecs)
        # return vecs
        
    def _normalize(self, vecs):
        """
        L2 Normalization with numerical stability.

        Normalizes vectors to unit length while preventing division by zero.
        """
        norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
        # Clip norms to prevent division by zero (min value: 1e-8)
        return vecs / np.clip(norms, 1e-8, np.inf)
        
    def _covariance(self, vecs):
        """
        Biased (1/N) covariance estimate, matching Eq. (2) of the LC-BERT paper.

        `np.cov` centers the data itself; `bias=True` selects the 1/N normalization
        rather than numpy's default 1/(N-1).
        """
        return np.cov(vecs.T, rowvar=True, bias=True)

    def _fix_signs(self, u):
        """
        Pin the sign of each eigenvector so the basis is reproducible.

        Eigenvectors are only defined up to a sign, and different LAPACK builds -
        or the same build on two different splits - return different column signs
        (Kessy et al. 2018, sec. 2). Setting diag(U) > 0 selects the unique
        transformation (ibid., sec. 5), which is what lets a kernel fitted on the
        training split be applied meaningfully to validation and test embeddings.
        """
        signs = np.sign(np.diag(u))
        signs[signs == 0] = 1.0
        return u * signs

    def _sorted_eigh(self, cov):
        """
        Symmetric eigendecomposition, eigenvalues descending, signs canonical.

        `np.linalg.eigh` guarantees real output for a symmetric matrix but returns
        eigenvalues in ASCENDING order. Both the LC-BERT paper (after Eq. 4) and
        Kessy et al. (sec. 2) assume descending order, which is also what makes
        truncation to the top `target_dim` components meaningful.
        """
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        return w, self._fix_signs(v)

    def _compute_kernel_bias_svd(self, vecs):
        """
        Calculate Kernel & Bias using SVD for whitening transformation.

        Transformation: y = (x + bias).dot(kernel)

        This method uses SVD to compute the whitening matrix that decorrelates
        the input features and normalizes their variance.

        Based on standard BERT-whitening approach (Su et al., 2021). Because the
        transformation is applied to row vectors, this kernel U*S^(-1/2) is the
        transpose of Kessy et al.'s W_PCA = S^(-1/2)*U^T, i.e. this branch already
        implements textbook PCA whitening. `np.linalg.svd` returns singular values
        in descending order, so the later top-k slice selects the leading components.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = self._covariance(vecs)
        u, s, vh = np.linalg.svd(cov)
        u = self._fix_signs(u)
        # Standard whitening: W = U * Σ^(-1/2)
        # No inverse needed - this is the correct formulation
        W = np.dot(u, np.diag(1.0 / np.sqrt(s + self.epsilon)))
        return W, -mu

    def _compute_kernel_bias_eigen(self, vecs):
        """
        Calculate Kernel & Bias using eigendecomposition.

        Similar to SVD method but uses eigendecomposition directly.
        Note: SVD is generally more numerically stable.

        `np.linalg.eig` (the general, non-symmetric solver) was returning unsorted
        and potentially complex eigenvalues for this symmetric covariance matrix;
        `_sorted_eigh` gives real values in descending order. The old
        `inv((U·Λ^(1/2))^T)` reduces to `U·Λ^(-1/2)`, so this branch is equivalent
        to `_compute_kernel_bias_svd` and is kept only for backwards compatibility.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = self._covariance(vecs)

        # Calculate Eigenvalues and Eigenvectors
        s, u = self._sorted_eigh(cov)
        s = np.maximum(s, self.epsilon)

        W = np.dot(u, np.diag(1.0 / np.sqrt(s + self.epsilon)))
        return W, -mu

    def _compute_kernel_bias_pca(self, vecs):
        """
        Calculate Kernel & Bias using PCA-based whitening via eigendecomposition.

        Kessy et al. (2018) Eq. (9) gives W_PCA = Λ^(-1/2)·V^T for the column-vector
        convention z = W·x. This pipeline applies the kernel to ROW vectors,
        y = (x + bias).dot(kernel), so the kernel must be the transpose:

            kernel = V·Λ^(-1/2)

        Returning Λ^(-1/2)·V^T here instead (as the paper's Eq. 5 does, transplanted
        from Kessy's column convention) violates the defining whitening constraint
        W^T·W = Σ^(-1): it yields Λ^(-1) rather than V·Λ^(-1)·V^T. Columns are ordered
        by descending eigenvalue, so slicing `kernel[:, :k]` keeps the top-k
        components - the compression optimality of Kessy et al.'s Proposition 3.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Covariance matrix estimation
        covariance_matrix = self._covariance(vecs)

        # Real eigenvalues, descending order, canonical eigenvector signs
        w, v = self._sorted_eigh(covariance_matrix)

        # This handles numerical precision issues
        w = np.maximum(w, self.epsilon)

        # Create diagonal matrix (now guaranteed to be real)
        diagw = np.diag(1.0 / np.sqrt(w + self.epsilon))

        # Row-vector convention: V·Λ^(-1/2), components down the columns
        pca_matrix = np.dot(v, diagw)

        return pca_matrix, -mu

    def _compute_kernel_bias_pca_cor(self, vecs):
        """
        Calculate Kernel & Bias using PCA-cor whitening (Kessy et al. 2018, Eq. 12).

        PCA-cor whitens the *standardized* variables: it rotates by the eigenmatrix
        of the CORRELATION matrix and scales by its eigenvalues, making the
        transformation scale-invariant. Kessy et al. recommend it over plain PCA
        whitening whenever the goal is maximal compression (their Proposition 4 and
        Conclusion) - which is the goal here, since the 768-dim embedding is
        truncated to `target_dim`.

        Column convention: W = Θ^(-1/2)·G^T·V^(-1/2). Transposed for row vectors:

            kernel = V^(-1/2)·G·Θ^(-1/2)

        where V = diag(σ²) holds the feature variances, P = V^(-1/2)·Σ·V^(-1/2) is the
        correlation matrix, and P = G·Θ·G^T its eigendecomposition.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        covariance_matrix = self._covariance(vecs)

        # Standardize: correlation matrix from the covariance matrix
        sd = np.sqrt(np.maximum(np.diag(covariance_matrix), self.epsilon))
        correlation_matrix = covariance_matrix / np.outer(sd, sd)

        # Eigendecomposition of the correlation matrix, descending, signs canonical
        theta, g = self._sorted_eigh(correlation_matrix)
        theta = np.maximum(theta, self.epsilon)

        diag_theta = np.diag(1.0 / np.sqrt(theta + self.epsilon))
        pca_cor_matrix = np.dot(np.dot(np.diag(1.0 / sd), g), diag_theta)

        return pca_cor_matrix, -mu

    def _compute_kernel_bias_pca_svd(self, vecs):
        """
        Calculate Kernel & Bias using PCA-based whitening via SVD.

        More numerically stable than eigendecomposition-based PCA.
        Uses SVD to compute the whitening transformation.

        Previously this built `np.diag(s)` and then called `np.diag` on that 2-D
        matrix, which EXTRACTS a diagonal rather than building one - collapsing the
        kernel to a 1-D array and making `kernel[:, :target_dim]` raise IndexError.
        The scaling is now built directly from the 1-D singular values, and the
        kernel is transposed into the row-vector convention (see
        `_compute_kernel_bias_pca`).
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Covariance matrix estimation
        covariance_matrix = self._covariance(vecs)

        _, s, vh = np.linalg.svd(covariance_matrix)
        v = self._fix_signs(vh.T)

        diag_sigma_inv = np.diag(1.0 / np.sqrt(s + self.epsilon))

        # Row-vector convention: V·Σ^(-1/2)
        pca_matrix = np.dot(v, diag_sigma_inv)

        return pca_matrix, -mu
    
    def _compute_kernel_bias_zca(self, vecs):
        """
        ZCA whitening using sklearn PCA (not currently used - kept for reference).
        Use _compute_kernel_bias_zca_base for standard ZCA whitening.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Center the data
        # centered_vecs = vecs - vecs.mean(axis=0, keepdims=True)

        # Apply PCA
        pca = PCA()
        # pca.fit(centered_vecs)
        pca.fit(vecs)

        # Calculate eigenvalues and eigenvectors
        # eigenvalues, eigenvectors = np.linalg.eig(pca.components_)
        w, v = np.linalg.eig(pca.components_)

        # Create a diagonal matrix with epsilon for numerical stability
        diagw = np.diag(1/((w + self.epsilon)**0.5))
        diagw = diagw.real.round(4)

        # Whitening transformation matrix
        zca_matrix = np.dot(np.dot(v, diagw), v.T)

        # Return ZCA whitening matrix and mean
        return zca_matrix, -mu
    
    def _compute_kernel_bias_zca_base(self, vecs):
        """
        Calculate Kernel & Bias using ZCA (Zero-phase Component Analysis) whitening.

        ZCA whitening: W = V * D^(-1/2) * V^T where V contains eigenvectors
        and D contains eigenvalues of the covariance matrix.

        Unlike PCA, ZCA preserves the structure of the original data better
        by rotating back to the original coordinate system.

        W_ZCA = Σ^(-1/2) is symmetric, so - unlike PCA - it is unaffected by the
        row/column convention, by eigenvalue ordering, and by eigenvector signs.
        Note the trade-off: Kessy et al.'s Proposition 1 shows ZCA is optimal for
        keeping the whitened variables SIMILAR to the originals, not for
        compression, so slicing `kernel[:, :k]` keeps the first k original
        embedding dimensions rather than the k most informative directions.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Covariance matrix estimation
        covariance_matrix = self._covariance(vecs)

        # Real eigenvalues, descending order, canonical eigenvector signs
        w, v = self._sorted_eigh(covariance_matrix)

        # This handles numerical precision issues
        w = np.maximum(w, self.epsilon)

        # Create diagonal matrix (now guaranteed to be real)
        diagw = np.diag(1.0 / np.sqrt(w + self.epsilon))

        # ZCA whitening transformation matrix: V * D^(-1/2) * V^T
        zca_matrix = np.dot(np.dot(v, diagw), v.T)

        return zca_matrix, -mu

    def _compute_kernel_bias_zca_svd(self, vecs):
        """
        Calculate Kernel & Bias using ZCA whitening via SVD.

        More numerically stable than eigendecomposition-based ZCA.
        Uses SVD to compute ZCA transformation: V * Σ^(-1/2) * V^T

        Carried the same `np.diag`-on-a-2-D-matrix defect as
        `_compute_kernel_bias_pca_svd`; the scaling is now built from the 1-D
        singular values. The product is symmetric, so no transpose is needed.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Covariance matrix estimation
        covariance_matrix = self._covariance(vecs)

        _, s, vh = np.linalg.svd(covariance_matrix)
        v = self._fix_signs(vh.T)

        diag_sigma_inv = np.diag(1.0 / np.sqrt(s + self.epsilon))

        # ZCA whitening transformation matrix: V·Σ^(-1/2)·V^T
        zca_matrix = np.dot(np.dot(v, diag_sigma_inv), v.T)

        return zca_matrix, -mu

    def _compute_kernel_bias_sphere(self, vecs):
        """
        Spherical whitening (not currently used - kept for reference).
        Simply normalizes vectors to unit length without decorrelation.
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Normalize data length
        normalized_vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]

        # Return whitening matrix (identity matrix in this case) and mean
        return np.eye(normalized_vecs.shape[1]), -mu

    def save_array(self, array, filename):
        """
        Save embeddings array to disk.

        Args:
            array: Array to save
            filename: Output filename (without extension)
        """
        vecs = np.array(array)
        np.save(f'{filename}.npy', vecs)

    def sent_to_vec(self, sent, pooling, max_len):
        """
        Convert a sentence to a vector representation using BERT/RoBERTa.

        Args:
            sent: Input sentence string
            pooling: Pooling strategy ('first_last_avg', 'last_avg', 'last2avg', 'cls')
            max_len: Maximum sequence length

        Returns:
            numpy array: Sentence embedding vector
        """
        with torch.no_grad():

            inputs = self.tokenizer.encode_plus(sent, return_tensors="pt", max_length=max_len,
                                               return_attention_mask=True, truncation=True)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

            hidden_states = self.model(input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        return_dict=True, output_hidden_states=True).hidden_states

            if pooling == 'first_last_avg':
                # Average of first and last hidden layers, then mean pool over tokens
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif pooling == 'last_avg':
                # Mean pool over tokens in last layer
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif pooling == 'last2avg':
                # Average of last two layers, then mean pool over tokens
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            elif pooling == 'cls':
                # Use [CLS] token from last layer
                output_hidden_state = (hidden_states[-1])[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling strategy: '{pooling}'. "
                               f"Supported: 'first_last_avg', 'last_avg', 'last2avg', 'cls'")


            vec = output_hidden_state.cpu().numpy()[0]

        return vec


    def Dim_reduction(self, max_len, dim_technique, pooling='first_last_avg', target_dim=256,
                      whitening_params=None):
        """
        Extract features using BERT/RoBERTa and apply dimensionality reduction.

        This method:
        1. Extracts embeddings from pretrained model for all sentences
        2. Applies selected whitening transformation
        3. Reduces dimensionality to target_dim
        4. Normalizes the output

        Args:
            max_len: Maximum sequence length for tokenization
            dim_technique: Whitening technique ('svd', 'eigen', 'zca', 'zca-svd', 'pca',
                'pca-svd', 'pca-cor')
            pooling: Pooling strategy for combining token embeddings (default: 'first_last_avg')
            target_dim: Target dimensionality after reduction (default: 256)
            whitening_params: Optional (kernel, bias) fitted on another split - normally
                the training split. When supplied, no kernel is fitted here and the
                supplied transformation is applied as-is, so all splits end up in the
                same basis. When None (the default) the kernel is fitted on this
                split's own embeddings, which is the original behaviour.

        Returns:
            numpy array: Whitened and dimensionally reduced embeddings
        """

        torch.cuda.empty_cache()

        if self.tokenizer is None or self.model is None:
            raise Exception("Tokenizer and Model must be defined for feature extraction")

        # Extract embeddings for all sentences
        vecs = []
        i = 1
        for sentence in self.data['cleaned_text']:
            vec = self.sent_to_vec(sentence, pooling, max_len)
            print(f'\rEmbedding Extraction: {i} / {len(self.data["cleaned_text"])} sentences | '
                  f'GPU Usage: {(torch.cuda.memory_allocated() / 1048576):.1f} MB', end=' ')
            i += 1
            vecs.append(vec)
        
        # Stack all embeddings into a single matrix
        # ✅ CHECK 1: Validate input embeddings
        embeddings = np.vstack(vecs)
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            raise ValueError("Input embeddings contain NaN or Inf values!")

        if whitening_params is not None:
            # Reuse the transformation fitted on another split (normally train), so
            # every split lands in the same basis. Only the embeddings above are
            # split-specific.
            kernel, bias = whitening_params
            print(f'\nReusing whitening kernel fitted on the training split: {kernel.shape}')
        else:
            # Compute kernel and bias based on selected technique
            if dim_technique == 'svd':
                kernel, bias = self._compute_kernel_bias_svd([vecs])
            elif dim_technique == 'eigen':
                kernel, bias = self._compute_kernel_bias_eigen([vecs])
            elif dim_technique == 'zca':
                kernel, bias = self._compute_kernel_bias_zca_base([vecs])
            elif dim_technique == 'zca-svd':
                kernel, bias = self._compute_kernel_bias_zca_svd([vecs])
            elif dim_technique == 'pca':
                kernel, bias = self._compute_kernel_bias_pca([vecs])
            elif dim_technique == 'pca-svd':
                kernel, bias = self._compute_kernel_bias_pca_svd([vecs])
            elif dim_technique == 'pca-cor':
                kernel, bias = self._compute_kernel_bias_pca_cor([vecs])
            else:
                raise ValueError(f"Unknown dimensionality reduction technique: '{dim_technique}'. "
                               f"Supported techniques: 'svd', 'eigen', 'zca', 'zca-svd', 'pca', "
                               f"'pca-svd', 'pca-cor'")

            # ✅ CHECK 2: Validate kernel and bias
            if np.isnan(kernel).any() or np.isinf(kernel).any():
                raise ValueError(f"Kernel contains NaN: {np.isnan(kernel).any()} /Inf: {np.isinf(kernel).any()} for technique '{dim_technique}'!")
            if np.isnan(bias).any() or np.isinf(bias).any():
                raise ValueError(f"Bias contains NaN: {np.isnan(bias).any()} /Inf: {np.isinf(bias).any()} for technique '{dim_technique}'!")

            # Reduce dimensionality by selecting top components
            kernel = kernel[:, :target_dim]

        # Expose the transformation so the next split can reuse it
        self.whitening_params = (kernel, bias)

        # Apply whitening transformation and normalize
        # This decorrelates features and makes covariance closer to identity matrix
        embeddings = self._transform_and_normalize(embeddings,
                    kernel=kernel,
                    bias=bias
                )
        
        # ✅ CHECK 3: Final validation
        # Validate output
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            raise ValueError(f"Output contains {nan_count} NaN and {inf_count} Inf values! "
                        f"Technique: '{dim_technique}'")
        
        print(f"\n✅ Whitening successful: {embeddings.shape}")
        return embeddings

class BertWhiteningDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super(BertWhiteningDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )