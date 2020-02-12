import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)

        return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # Fill dists[i_test][i_train]
                dists[i_test][i_train] = np.absolute(self.train_X[i_train] - X[i_test]).sum()              
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dists[i_test] = np.absolute(self.train_X - X[i_test]).sum(axis = 1).T 
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # Implement computing all distances with no loops!
        dists = np.absolute(self.train_X[:, np.newaxis] - X).sum(axis=2).T
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # Implement choosing best class based on k nearest training samples
            k_th_element = np.partition(dists[i], self.k)[self.k - 1]
            cluster_indexes = np.where(dists[i] <= k_th_element)[0]
            # --delete k_th element repetitions 
            if cluster_indexes.size > self.k:
                cluster_indexes = np.delete(cluster_indexes, np.in1d(cluster_indexes, \
                        np.where(dists[i] == k_th_element)[:(cluster_indexes.size - self.k)]).nonzero()[0])
            # --train_y here is array of bool values
            cluster = self.train_y[cluster_indexes]
            if (cluster * 1).sum() > 0.5 * cluster.size:
                pred[i] = True
            else:
                pred[i] = False
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # Implement choosing best class based on k nearest training samples
            k_th_element = np.partition(dists[i], self.k)[self.k - 1]
            cluster_indexes = np.where(dists[i] <= k_th_element)[0]
            # --delete k_th element repetitions
            if cluster_indexes.size != self.k:
                cluster_indexes = np.delete(cluster_indexes, np.in1d(cluster_indexes, \
                        np.where(dists[i] == k_th_element)[:(cluster_indexes.size - self.k)]).nonzero()[0])
            # --create sorted array of integer occurrences counts
            counts = np.bincount(self.train_y[cluster_indexes])
            # --find index of maximum value and check if it is determined number
            pred[i] = np.argmax(counts)
        return pred
