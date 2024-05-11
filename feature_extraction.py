from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA, NMF
# from bertopic import BERTopic
import numpy as np
import torch
import torch.nn as nn

class FeatureExtraction:
    def __init__(self, corpus, tokenizer=None, model=None, device=torch.device('cpu')):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    #Defining a function to extract feature using LDA 
    def lda_feature_extraction(self, num_topics=10):
    
        # preprocess data
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.corpus)
        lda = LatentDirichletAllocation(n_components=num_topics)
        X_topics = lda.fit_transform(X)
        return X_topics
    
    #Defining a function to extract feature using BertTopic
    def bertopic_feature_extraction(self, num_topics=10):
    
        # Train a BERTopic model
        topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
        topics, probs = topic_model.fit_transform(self.corpus)
        word_probs = [np.array([probs for (word, probs) in topic_model.get_topic(topic)]) for topic in topics]
         
        return np.array(word_probs)

    def _transform_and_normalize(self, vecs, kernel, bias):
        """
            Applying transformation then standardize
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return self._normalize(vecs)
        
    def _normalize(self, vecs):
        """
            Standardization
        """
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
        
    def _compute_kernel_bias(self, vecs):
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

    #Defining a function to extract feature using Bert and Dimensionality Reduction
    def Dim_reduction(self, max_len, target_dim=256):
        '''
            This method will accept array of sentences, roberta tokenizer & model
            next it will call methods for dimention reduction
        '''
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
    
        vecs = []
        with torch.no_grad():
          i = 1
          for sentence in self.corpus:
            inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

            hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True, output_hidden_states=True).hidden_states

            #Averaging the first & last hidden states
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)

            vec = output_hidden_state.cpu().numpy()[0]

            print(f'\rWord Embedding Process: {i} words | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
            i+=1

            vecs.append(vec)
    
        #Finding Kernal
        kernel, bias = self._compute_kernel_bias([vecs])
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
        
    #Defining a function to extract feature using Bert and Dimensionality Reduction
    def Dim_reduction_long_document(self, max_len, target_dim=256):
        '''
            This method will accept array of sentences, roberta tokenizer & model
            next it will call methods for dimention reduction
        '''
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
    
        vecs = []
        with torch.no_grad():
          i = 1
          for sentence in self.corpus:
            inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True)
            
            num_tokens = inputs.input_ids.size(1)
            
             # check sequence token limit
            if num_tokens <= fragment_size:
                inputs['input_ids'] = inputs['input_ids'].to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
    
                hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True, output_hidden_states=True).hidden_states
    
                #Averaging the first & last hidden states
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
    
                vec = output_hidden_state.cpu().numpy()[0]
                
            else:
                fragment_embeddings = []
                # Slide a window over the document to capture context
                for j in range(0, num_tokens, fragment_size):
                    start = j
                    end = min(j + fragment_size, num_tokens)

                    fragment_input_ids = tokenized_document.input_ids[0, start:end].unsqueeze(0).to(self.device)
                    fragment_attention_mask = tokenized_document.attention_mask[0, start:end].unsqueeze(0).to(self.device)

                    hidden_states = self.model(input_ids=fragment_input_ids, attention_mask=fragment_attention_mask,
                                            output_hidden_states=True)[0]
                                            
                    #Averaging the first & last hidden states
                    output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
                    embedding = output_hidden_state.cpu().numpy()[0]

                    fragment_embeddings.append(embedding)
                    
                # Average fragment embeddings to get document embedding
                avg_doc_embedding = np.mean(fragment_embeddings, axis=0)
                doc_embeddings.append(avg_doc_embedding)
                
            print(f'\rWord Embedding Process: {i} words | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
            i+=1

            vecs.append(vec)
    
        #Finding Kernal
        kernel, bias = self._compute_kernel_bias([vecs])
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
    
    #Defining a function to extract feature using Bert
    def get_embeddings(self, max_len):
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
    
        vecs = []
        with torch.no_grad():
          i = 1
          for sentence in self.corpus:
              inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
              inputs['input_ids'] = inputs['input_ids'].to(self.device)
              inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
    
              hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)[0]
              embedding = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
              vec = embedding.cpu().numpy()[0]
    
              vecs.append(vec)
              print(f'\rWord Embedding Process: {i} words | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
              i+=1
    
        return np.array(vecs)

    def get_embeddings_long_document(self, max_len, fragment_size):
        torch.cuda.empty_cache()

        if self.tokenizer is None or self.model is None:
            raise Exception("Sorry, but you must define Tokenizer and Model")

        doc_embeddings = []
        with torch.no_grad():
            i = 1
            for document in self.corpus:
                tokenized_document = self.tokenizer.encode_plus(document, return_tensors="pt", max_length=max_len,
                                                    return_attention_mask=True, padding='max_length')

                num_tokens = tokenized_document.input_ids.size(1)
                
                # check sequence token limit
                if num_tokens <= fragment_size:
                    input_ids = tokenized_document.input_ids.to(self.device)
                    attention_mask = tokenized_document.attention_mask.to(self.device)
                    hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                            output_hidden_states=True)[0]
                    embedding = hidden_states[:, 0, :].cpu().numpy()
                    doc_embeddings.append(embedding)
                else:
                    fragment_embeddings = []
                    # Slide a window over the document to capture context
                    for j in range(0, num_tokens, fragment_size):
                        start = j
                        end = min(j + fragment_size, num_tokens)
    
                        fragment_input_ids = tokenized_document.input_ids[0, start:end].unsqueeze(0).to(self.device)
                        fragment_attention_mask = tokenized_document.attention_mask[0, start:end].unsqueeze(0).to(self.device)
    
                        hidden_states = self.model(input_ids=fragment_input_ids, attention_mask=fragment_attention_mask,
                                                output_hidden_states=True)[0]
                        embedding = hidden_states[:, 0, :].cpu().numpy()
    
                        fragment_embeddings.append(embedding)
    
                    # Average fragment embeddings to get document embedding
                    avg_doc_embedding = np.mean(fragment_embeddings, axis=0)
                    doc_embeddings.append(avg_doc_embedding)
                print(f'\rDocument Embedding Process: {i} documents | GPU Usage: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
                i += 1

        return np.array(doc_embeddings)
    
    #Defining a function to extract feature using Bert and dim reduction using LDA
    def dim_reduction_lda(self, max_len, reduction_size):
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
            
        lda = LatentDirichletAllocation(n_components=reduction_size)
    
        vecs = []
        with torch.no_grad():
          i = 1
          for sentence in self.corpus:
              inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
              inputs['input_ids'] = inputs['input_ids'].to(self.device)
              inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
    
              hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)[0]
              embedding = hidden_states[:, 0, :]
              vec = embedding
              
              m = nn.Softplus()
              soft_vec = m(vec)
              dim_reduction_vec = lda.fit_transform(soft_vec.cpu().numpy())
              
              vecs.append(dim_reduction_vec)
              print(f'\rWord Embedding Process: {i} words | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
              i+=1
    
        return np.array(vecs)
        
    #Defining a function to extract feature using Bert and dim reduction using LDA
    def dim_reduction_nmf(self, max_len, reduction_size):
        
        torch.cuda.empty_cache()
        
        if(self.tokenizer is None or self.model is None):
            raise Exception("Sorry, But You must define Tokenizer and Model")
            
        nmf = NMF(n_components=reduction_size)
    
        vecs = []
        with torch.no_grad():
          i = 1
          for sentence in self.corpus:
              inputs = self.tokenizer.encode_plus(sentence, return_tensors="pt",max_length=max_len, return_attention_mask=True,truncation=True)
              inputs['input_ids'] = inputs['input_ids'].to(self.device)
              inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
    
              hidden_states = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)[0]
              embedding = hidden_states[:, 0, :]
              vec = embedding
              
            #   m = nn.Softplus()
            #   soft_vec = m(vec)
              dim_reduction_vec = nmf.fit_transform(vec.cpu().numpy())
              
              vecs.append(dim_reduction_vec)
              print(f'\rWord Embedding Process: {i} words | GPU Usages: {(torch.cuda.memory_allocated() / 1048576):3.1f}', end=' ')
              i+=1
    
        return np.array(vecs)