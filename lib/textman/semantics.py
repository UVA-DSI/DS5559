import pandas as pd
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

class Semantic:
    
    def  __init__(self, WVM):
        self.WVM = WVM # Test to see if this really is a WM

    def get_word_vector(self, term_str):
        """Get a numpy array from the glove matrix and shape for input into cosine function"""
        wv = self.WVM.loc[term_str].values.reshape(-1, 1).T
        return wv

    def get_sims(self, term_str, n=10):
        """Get the top n words for a given word based on cosine similarity"""
        wv = get_word_vector(term_str)
        sims = cosine_similarity(self.WVM.values, wv)
        return pd.DataFrame(sims, index=self.WVM.index, 
                            columns=['score']).sort_values('score', ascending=False).head(n)

    def get_nearest_vector(self, wv, method='cosine'):
        """Get the nearest word vector to a given word vector"""
        if method == 'cosine':
            sims = cosine_similarity(self.WVM.values, wv)
        elif method == 'euclidean':
            sims = euclidean_distances(self.WVM.values, wv)
        else:
            print('Invalid method {}; defaulting to cosine.'.format(method))
            sims = cosine_similarity(self.WVM.values, wv)
        return pd.DataFrame(sims, index=self.WVM.index, 
                            columns=['score']).sort_values('score',ascending=False).head(2).iloc[1]

    def get_analogy(a, b, d, method='cosine'):
        """Infer missing analogical term"""
        try:
            A = get_word_vector(a)
            B = get_word_vector(b)
            D = get_word_vector(d)
            C = np.add(np.subtract(A, B), D)
            X = get_nearest_vector(C, method=method)
            return X.name
        except ValueError as e:
            print(e)
            return None

        
if __name__  == '__main__':
    pass