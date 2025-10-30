import numpy as np

class SVM:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def _init_weights_bias(self, X: np.array):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y: np.array):
        """
        We need to convert {1,0} to {1,-1} which enables us to decide on which side does the feature lie
        To compute that we can use the numpy's built in function -> np.where
        """
        return np.where(y<=0, -1, 1)


    # Check if each single points satisfies the constraint or not
    def _satisfy_constraints(self, x:np.array, idx:int):
        linear_model = np.dot(x, self.w) + self.b

        # the cls map contains 1s and -1s,
        # The linear model >= 1 checks on which side does this point lie in the plane
        return self.cls_map[idx] * linear_model >= 1

    def _get_gradients(self, constrain, x, idx):
        """
        Correct side: 
        dw = lanbda * weights
        db = 0

        Wrong side:
        dw = lambda * weights - y(self.cls_map) dot x
        db = -self.cls_map
        """

        # Condition if the data point lies on the correct side
        # See the main objective function on notes for more mathematical info
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        
        # If on the wrong side of the hyperplane
        dw = (self.lambda_param * self.w) - (self.cls_map[idx] * x)
        db = -self.cls_map[idx]

        return dw, db

    def _update_weights_bias(self, dw, db):
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
    
    def fit(self, X:np.array, y:np.array):
        # initilize the matrices
        self._init_weights_bias(X)

        # Generate class maps 
        self.cls_map = self._get_cls_map(y)

        # Perform the epochs to check calculate and updated weights and biases
        for i in range(self.n_iters):
            for idx, x in enumerate(X):
                # Check the contraint
                constraint = self._satisfy_constraints(x, idx)

                # Compute the gradients
                dw, db = self._get_gradients(constraint, x, idx)

                # Update the gradients
                self._update_weights_bias(dw, db)
                
                # * Some text From medium blog
                # In order to compute the correct gradients, we need to know which case of the objective function
                # is relevant with respect to a specific data point. Hence, we need to check
                # if the data point satisfies the constraint
                # yi(w.x + b) >= 1


    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        # compute the sign
        prediction = np.sign(estimate)
        # map class from {-1, 1} to original values {0, 1}
        return np.where(prediction == -1, 0, 1)



