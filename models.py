import numpy as np
from scipy.sparse import csr_matrix, special


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

class LambdaMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']
        
        self.num_examples, self.num_features = X.shape  
        
        X = np.array(X.todense())
        X_T = np.array(np.transpose(X)) # transpose for column access
        
        self.prototypes = []
        self.prototypes.append([i.mean() for i in X_T])
        self.prototypes = np.transpose(self.prototypes)
        
        if (lambda0 == 0):
            lambda0 = np.mean([np.linalg.norm(self.prototypes[:,0]-i) for i in X])
        
        for _ in range(iterations):
            r_nk = np.zeros([self.num_examples,self.prototypes.shape[1]])
            for i in range(self.num_examples):
                dist = np.array([np.linalg.norm(self.prototypes[:,j]-X[i]) for j in range(self.prototypes.shape[1])])
                if all(dist > lambda0):
                    self.prototypes = np.concatenate((self.prototypes,np.expand_dims(X[i],axis=1)), axis=1)
                    r_nk = np.concatenate((r_nk,np.zeros([self.num_examples,1])), axis=1)
                    r_nk[i,(r_nk.shape[1]-1)] = 1
                else:
                    r_nk[i,np.argmin(dist)] = 1
                    
            #M step
            temp_prototypes = []
            for i in range(self.prototypes.shape[1]):
                r = np.expand_dims(r_nk[:,i], axis=1)
                temp_prototype_vector = [np.sum((X*r)[:,j])/np.sum(r) for j in range(self.num_features)]
                temp_prototypes.append(temp_prototype_vector)
            self.prototypes = np.transpose(np.array(temp_prototypes))
                

    def predict(self, X):
        X = np.array(X.todense())
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dist = np.array([np.linalg.norm(self.prototypes[:,j]-X[i]) for j in range(self.prototypes.shape[1])])
            y_hat[i] = np.argmin(dist)
        
        return y_hat

class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.
        
        self.num_examples, self.num_features = X.shape  
        
        X = np.array(X.todense())
        X_T = np.array(np.transpose(X)) # transpose for column access
        
        K = num_clusters
        c = 2
        
        self.prototypes = []
        
        if (K == 1):
            self.prototypes.append([i.mean() for i in X_T])
        else:
            prototype_min = np.array([i.min() for i in X_T])
            prototype_max = np.array([i.max() for i in X_T])
            
            self.prototypes.append(list(prototype_min))
            dif_incr = (prototype_max-prototype_min)/(K-1)
            for i in range(1,K):
                self.prototypes.append(list(prototype_min+(i*dif_incr)))
            
        self.prototypes = np.transpose(self.prototypes)
        
        for _ in range(iterations):
            beta = c * (_ + 1)
            p_nk = np.zeros([self.num_examples,self.prototypes.shape[1]])
            for i in range(self.num_examples):
                d_hat = np.mean([np.linalg.norm(self.prototypes[:,j] - X[i]) for j in range(self.prototypes.shape[1])])
                denom = np.sum([np.exp(((-beta)*np.linalg.norm(self.prototypes[:,j]-X[i]))/d_hat) for j in range(self.prototypes.shape[1])])
                p_nk[i] = np.array([(np.exp(((-beta)*np.linalg.norm(self.prototypes[:,j]-X[i]))/d_hat))/denom for j in range(self.prototypes.shape[1])])
        
            #M step
            temp_prototypes = []
            for i in range(self.prototypes.shape[1]):
                p = np.expand_dims(p_nk[:,i], axis=1)
                temp_prototype_vector = [np.sum((X*p)[:,j])/np.sum(p) for j in range(self.num_features)]
                temp_prototypes.append(temp_prototype_vector)
            self.prototypes = np.transpose(np.array(temp_prototypes))
            

    def predict(self, X):
        X = np.array(X.todense())
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dist = np.array([np.linalg.norm(self.prototypes[:,j]-X[i]) for j in range(self.prototypes.shape[1])])
            y_hat[i] = np.argmin(dist)
        
        return y_hat

class LogisticRegression(Model):

    def __init__(self, learning_rate, num_selected_features, gd_iterations):
        super().__init__()
        self.w = 0;
        self.learning_rate = learning_rate
        self.num_selected_features = num_selected_features
        self.gd_iterations = gd_iterations
        self.feature_list = None
    
    def selectfeatures(self, X, y):
        H_list = []
        for i in range(self.num_input_features):
            thresh = np.mean(X[:,i])
            above_thresh_correct, above_thresh_incorrect, below_thresh_correct,below_thresh_incorrect = 0,0,0,0
            for j in range(self.num_examples):
                if (X[j,i] >= thresh):
                    if (y[j] == 1): above_thresh_correct += 1
                    else: above_thresh_incorrect += 1
                else:
                    if (y[j] == 1): below_thresh_incorrect += 1
                    else: below_thresh_correct += 1
            above = above_thresh_correct + above_thresh_incorrect
            below = below_thresh_correct + below_thresh_incorrect
            total = above + below
            H_above = 0 if (above == 0 or above_thresh_correct == 0 or above_thresh_incorrect == 0) else (-(above_thresh_correct/above)*np.log(above_thresh_correct/above)) - ((above_thresh_incorrect/above)*np.log(above_thresh_incorrect/above))
            H_below = 0 if (below == 0 or below_thresh_correct == 0 or below_thresh_incorrect == 0) else (-(below_thresh_correct/below)*np.log(below_thresh_correct/below)) - ((below_thresh_incorrect/below)*np.log(below_thresh_incorrect/below))        
            H_Y_X = (above/total)*H_above + (below/total)*H_below
            H_list.append((H_Y_X, i))
            
        H_list = sorted(H_list)
        self.feature_list = sorted([i[1] for i in H_list][0:10])
        ret = np.zeros([self.num_examples,self.num_selected_features],dtype=np.double)
        for i in range(self.num_selected_features):
            ret[:,i] = X[:,self.feature_list[i]]
        return ret                  

    def fit(self, X, y):
        self.num_examples, self.num_input_features = X.shape
        X = np.asarray(X.todense())
        
        if (self.num_selected_features != -1): X = self.selectfeatures(X,y)
        else: self.num_selected_features = self.num_input_features
            
        self.w = np.zeros(self.num_selected_features,dtype=np.double)
        for i in range(self.gd_iterations):
            gd = 0
            for j in range(self.num_examples):
                y_hat = special.expit(np.dot(self.w,X[j]))
                y_hat_neg = special.expit(-1*np.dot((self.w),X[j]))
                gd += (y[j]*y_hat_neg*X[j]) + ((1-y[j])*y_hat*(-X[j]))
            self.w = self.w + self.learning_rate*(gd)
        

    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        
        X = np.asarray(X.todense())
        if(self.feature_list is not None): X = X[:, self.feature_list]  
        
        y_hat = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            y_hat[i] = round(special.expit(np.dot(self.w,X[i])))
                
        return y_hat



class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        X = np.asarray(X.todense())
        
        y_hat = np.zeros(X.shape[0], dtype=int)
        label_size = X.shape[1]
        first_bound = label_size // 2
        second_bound = label_size - first_bound
        
        for i in range(X.shape[0]):
            first_sum = 0
            second_sum = 0
            for k in range(first_bound):
                first_sum += X[i][k]
            for k in range(second_bound, label_size):
                second_sum += X[i][k]
                
            if first_sum >= second_sum:
                y_hat[i] = 1
                
        return y_hat


class Perceptron(Model):

    def __init__(self, learning_rate, iterations):
        super().__init__()
        self.w = None
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.num_examples, self.num_input_features = X.shape
        
        for i in range(len(y)):
            if (y[i] == 0):
                y[i] = -1
        
        X = np.asarray(X.todense())
        
        self.w = np.zeros(self.num_input_features, dtype=int)
        for k in range(self.iterations):
            for i in range(self.num_examples):
                y_hat = int(np.sign(np.dot(self.w,X[i])))
                y_hat = 1 if y_hat >= 0 else -1
                self.w = self.w + (self.learning_rate*(y[i]-y_hat)*X[i])

    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        
        X = np.asarray(X.todense())
        y_hat = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            y_hat[i] = int(np.sign(np.dot(self.w,X[i])))
            y_hat[i] = 1 if y_hat[i] >= 0 else 0
                
        return y_hat
    
class Adaboost(Model):

    def __init__(self, iterations):
        super().__init__()
        self.T = iterations
        self.error_thresh =  0.000001
        
    def convert_labels(self, y):
        if (-1 not in y):
            y = (y-1) + y
        return y
    
    def get_h(self, X, y, D, cutoffs):
        
        c, a, f, e = cutoffs[0][0], 0, 0, 999 
        for i in range(self.num_features):
            for j in range(cutoffs[i].shape[0]):
                cur_cutoff = cutoffs[i][j]
                x = X.getcol(i).toarray().flatten()
                h = (((x > cur_cutoff) & (y == -1)) | ((x <= cur_cutoff) & (y == 1))).astype(float)
                epsilon = np.dot(D,h)
                if (epsilon < e):
                    c, a, f, e = cur_cutoff, 0, i, epsilon
                if (epsilon < 0.000001):
                    return c,a,f,epsilon
                h = (((x <= cur_cutoff) & (y == -1)) | ((x > cur_cutoff) & (y == 1))).astype(float)
                epsilon = np.dot(D,h)
                if (epsilon < e):
                    c, a, f, e = cur_cutoff, 1, i, epsilon
                if (epsilon < 0.000001):
                    return c,a,f,epsilon
        return c,a,f,e
    
    def get_cutoff_matrix(self, X):
        cutoffs = []
        for i in range(self.num_features):
            x = np.sort(np.unique(X.getcol(i).toarray().squeeze())).tolist() #should this be unique?
            x = ((np.asarray((x+[0])) - np.asarray(([0]+x)))/2)+np.asarray([0]+x)
            x = x[1:(x.shape[0]-1)]
            cutoffs.append(np.unique(x))
        return cutoffs

    def fit(self, X, y):
        self.num_examples, self.num_features = X.shape
        y = self.convert_labels(y)
        
        self.D = np.full([self.num_examples], (1/self.num_examples))
        
        cutoffs = self.get_cutoff_matrix(X) #replace self in cutoffs
        self.H_c = [] #cutoff
        self.H_a = [] #boolean (ge or l)
        self.H_f = [] #feature idx
        self.alpha = []
        
        for i in range(self.T):
            c, a, f, e = self.get_h(X,y,self.D,cutoffs)
            if (e < 0.000001):
                break
            self.H_c.append(c)
            self.H_a.append(a)
            self.H_f.append(int(f))
            self.alpha.append(((0.5)*np.log((1-e)/e)))
            x = X.getcol(self.H_f[i]).toarray().flatten()
            if (a & 1):
                h = (x <= c).astype(float)
                h = (h-1) + h
            else:
                h = (x > c).astype(float)
                h = (h-1) + h
            Z = np.dot(self.D, np.exp((-self.alpha[i])*(y*h)))
            self.D = (1/Z)*(self.D*np.exp((-self.alpha[i])*(y*h)))
            
    def predict(self, X):
        
        examples, features = X.shape
        y_hat = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            x = X.getrow(i).toarray().flatten().tolist()
            idx = (np.array([x[j] for j in self.H_f]) > np.array(self.H_c)).astype(int)
            idx = np.abs(idx-np.asarray(self.H_a)).astype(int)
            idx = (idx-1) + idx
            idx = idx * np.asarray(self.alpha)
            y_hat[i] = 1 if (np.sum(idx) >= 0) else 0            
        
        return y_hat

class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


# TODO: Add other Models as necessary.
