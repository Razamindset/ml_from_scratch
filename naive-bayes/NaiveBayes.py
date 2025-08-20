import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate the mean variancce and prior for each class
        # Note: These vlaues are for each class over all data. Say red is a class then we wanna go over all data wherever the label is red

        # Mean has the size 2 X 1
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._prior = np.zeros(n_classes) # Similarly we will have  prior value for each class hence the size of rows is n_classes

        for idx, c in enumerate(self._classes):
            # X_c is the feature data for the current classs
            X_c = X[y == c]

            # ie say for data where label is red it will mean the values along the x axis or sum all the features
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)

            # Total frequency  = total data that has this feauture / Total data that exists, X_c.shape[0] gives the totl occurence of red
            #  X_c.shape[0] = [2,2][0]
            self._prior[idx] = X_c.shape[0] / float(n_samples)

    def _predict(self, x):
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior)
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
        


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))