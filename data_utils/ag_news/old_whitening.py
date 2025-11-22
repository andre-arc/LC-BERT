from pathlib import Path
import sys, os

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

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

    def __init__(self, device, dataset, tokenizer, model, max_len, dim_technique, *args, **kwargs):
        self.device = device
        self.data = self.load_dataset(dataset)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.vecs = self.Dim_reduction(dim_technique=dim_technique, max_len=max_len)
        
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
            Standardization
        """
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
        
    def _compute_kernel_bias_svd(self, vecs):
        """
        Calculate Kernal & Bias for the final transformation - y = (x + bias).dot(kernel)
        """
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(s**0.5))
        W = np.linalg.inv(W.T)
        return W, -mu

    def _compute_kernel_bias_eigen(self, vecs):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)

        # Calculate Eigenvalues and Eigenvectors
        u, s = np.linalg.eig(cov)

        W = np.dot(u, np.diag(s**0.5))
        W = np.linalg.inv(W.T)
        return W, -mu

    def _compute_kernel_bias_pca(self, vecs):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        
        # Center the data
        # centered_vecs = vecs - vecs.mean(axis=0, keepdims=True)
        
        # Covariance matrix estimation
        # covariance_matrix = np.cov(centered_vecs.T, rowvar=True, bias=True)
        covariance_matrix = np.cov(vecs.T, rowvar=True, bias=True)

        # Calculate Eigenvalues and Eigenvectors
        w, v = np.linalg.eig(covariance_matrix)

        # Create a diagonal matrix
        diagw = np.diag(1/((w+.1e-5)**0.5))
        diagw = diagw.real.round(4)

        pca_matrix = np.dot(diagw, v.T)

        # Invert and transpose W
        # W = np.linalg.inv(W.T)
        return pca_matrix, -mu

    def _compute_kernel_bias_pca_svd(self, vecs):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        
        # Center the data
        # centered_vecs = vecs - vecs.mean(axis=0, keepdims=True)
        
        # Covariance matrix estimation
        # covariance_matrix = np.cov(centered_vecs.T, rowvar=True, bias=True)
        covariance_matrix = np.cov(vecs.T, rowvar=True, bias=True)

        u, s, vh = np.linalg.svd(covariance_matrix)

        diag_sigma = np.diag(s)
        diag_sigma_inv = np.diag(1 / (diag_sigma**0.5 + 1e-5))

        pca_matrix = np.dot(diag_sigma_inv, vh.T)

        # Invert and transpose W
        # W = np.linalg.inv(W.T)
        return pca_matrix, -mu
    
    def _compute_kernel_bias_zca(self, vecs):
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

        # Create a diagonal matrix
        diagw = np.diag(1/((w)**0.5))
        diagw = diagw.real.round(4)

        # Whitening transformation matrix
        zca_matrix = np.dot(np.dot(v, diagw), v.T)

        # Return ZCA whitening matrix and mean
        return zca_matrix, -mu
    
    def _compute_kernel_bias_zca_base(self, vecs):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        
        # Center the data
        # centered_vecs = vecs - vecs.mean(axis=0, keepdims=True)
        
        # Covariance matrix estimation
        covariance_matrix = np.cov(vecs.T, rowvar=True, bias=True)

        # Calculate Eigenvalues and Eigenvectors
        w, v = np.linalg.eig(covariance_matrix)

        # Create a diagonal matrix
        diagw = np.diag(1/((w+.1e-5)**0.5))
        diagw = diagw.real.round(4)

        # Whitening transformation matrix
        zca_matrix = np.dot(np.dot(v, diagw), v.T)

        # Return ZCA whitening matrix and mean
        return zca_matrix, -mu

    def _compute_kernel_bias_zca_svd(self, vecs):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        
        # Center the data
        # centered_vecs = vecs - vecs.mean(axis=0, keepdims=True)
        
        # Covariance matrix estimation
        # covariance_matrix = np.cov(centered_vecs.T, rowvar=True, bias=True)
        covariance_matrix = np.cov(vecs.T, rowvar=True, bias=True)

        u, s, vh = np.linalg.svd(covariance_matrix)

        diag_sigma = np.diag(s)
        diag_sigma_inv = np.diag(1 / (diag_sigma**0.5 + 1e-5))

        # Whitening transformation matrix
        zca_matrix = np.dot(np.dot(vh, diag_sigma_inv), vh.T)

        # Return ZCA whitening matrix and mean
        return zca_matrix, -mu

    def _compute_kernel_bias_sphere(self, vecs):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)

        # Normalize data length
        normalized_vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]

        # Return whitening matrix (identity matrix in this case) and mean
        return np.eye(normalized_vecs.shape[1]), -mu

    def save_array(self, array, filename):
        vecs = np.array(array)
        # vecs.save(f'{filename}.npy', vecs)
        np.save(f'{filename}.npy', vecs)

    def sent_to_vec(self, sent, pooling, max_len):
        with torch.no_grad():
            
            inputs = self.tokenizer.encode_plus(sent, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

            hidden_states = self.model(input_ids=inputs['input_ids'], 
                                        attention_mask=inputs['attention_mask'], 
                                        return_dict=True, output_hidden_states=True).hidden_states

            if pooling == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif pooling == 'last_avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif pooling == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            elif pooling == 'cls':
                output_hidden_state = (hidden_states[-1])[:, 0, :]
            else:
                raise Exception("unknown pooling {}".format(pooling))
            

            vec = output_hidden_state.cpu().numpy()[0]

        return vec


    #Defining a function to extract feature using Bert and Dimensionality Reduction
    def Dim_reduction(self, max_len, dim_technique, pooling='first_last_avg', target_dim=256):
        '''
            This method will accept array of sentences, roberta tokenizer & model
            next it will call methods for dimention reduction
        '''
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
    
        vecs = []
        i = 1
        for sentence in self.data['cleaned_text']:
            vec = self.sent_to_vec(sentence, pooling, max_len)
            print(f'\rWord Embedding Process: {i} / {len(self.data["cleaned_text"])} words | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
            i+=1
            # print(vec.shape)
            vecs.append(vec)

        #Finding Kernal

        if(dim_technique == 'svd'):
            kernel, bias = self._compute_kernel_bias_svd([vecs])
        if(dim_technique == 'eigen'):
            kernel, bias = self._compute_kernel_bias_eigen([vecs])
        elif(dim_technique == 'zca'):
            kernel, bias = self._compute_kernel_bias_zca_base([vecs])
        elif(dim_technique == 'zca-svd'):
            kernel, bias = self._compute_kernel_bias_zca_svd([vecs])
        elif(dim_technique == 'pca'):
            kernel, bias = self._compute_kernel_bias_pca([vecs])
        elif(dim_technique == 'pca-svd'):
            kernel, bias = self._compute_kernel_bias_pca_svd([vecs])
            
        kernel = kernel[:, :target_dim]
        
        #If you want to reduce it to 128 dim
        #kernel = kernel[:, :128]
        embeddings = []
        embeddings = np.vstack(vecs)
    
        #Sentence embeddings can be converted into an identity matrix
        #by utilizing the transformation matrix
        embeddings = self._transform_and_normalize(embeddings, 
                    kernel=kernel,
                    bias=bias
                )
    
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