import numpy as np

class knn(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        """
        Just memorize the training data
        Inputs:
        - X is a numpy array with shape (n, D) containing the training data
        - y is a numpy 1d array containing the labels of the training data. y[i] is the label for X[i,:]
        """

        self.Xtr = X
        self.ytr = y
    
    def predict(self, X, k):
       """
       Predicts a classification for data inputs
       Inputs: 
       - X is a numpy array with shape (n, D) containing data for which we want to predict a label
       """

       distances = self.compute_distances(X)

       return self.predict_labels(distances, k)

    def compute_distances(self, Xtest):
        """
        Calculates Euclidean distance between test points and training points and returns
        a numpy array containing those distances
        Inputs:
        - X is a numpy array with shape (n, D) containing data for which we want to predict a label
        Outputs:
        - distances is a numpy array with shape (num_train, num_test) where num train is the number
        of training points and num_test is the number of test points and distance[i,j] is 
        the Euclidean distance between the ith training point and the jth testing point

        """
        num_train = self.Xtr.shape[0]
        num_test = Xtest.shape[0]

        distances = np.zeros((num_train, num_test))

        for i in range(num_train):
            for j in range(num_test):
                distances[i,j] = np.sqrt(np.sum((Xtest[j, :] - self.Xtr[i, :])**2))
        
        return distances

    def predict_labels(self, distances, k):
        """
        Generates an array containing predicted labels using most common label in the k nearest
        neighbouring data points to the input data points

        Inputs:
        - distances is a numpy array with shape (num_train, num_test) where num train is the number
        of training points and num_test is the number of test points and distance[i,j] is 
        the Euclidean distance between the ith training point and the jth testing point
        - k is the number of nearest neighbouring data points from the training set to be used 
        for comparison to the input data points

        Outputs:
        - y_predicted is a numpy 1d array with length num_test containing the predicted labels
        for the input data points where y_predicted[i] is the predicted label for Xtest[i] 
        """
        num_test = distances.shape[1]
        y_predicted = np.zeros(num_test)

        for i in range(num_test):
            nearest_neighbour_indices = np.argsort(distances[i, :][:k]).tolist() #gets nearest neighbour indices by sorting then slicing
            nearest_neighbour_labels = self.ytr[nearest_neighbour_indices] #grabs labels for those indices

            label_count = np.bincount(nearest_neighbour_labels)

            y_predicted[i] = np.argmax(label_count) 
        
        return y_predicted