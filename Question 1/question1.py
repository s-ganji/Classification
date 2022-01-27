import numpy as np

class regression:

    def sigmoid(self,result):
        result_new = 1 / (1 + np.exp(-result))
        return result_new

    def model(self,w, b, X, Y):
        # Prediction and sigmoid
        result = self.sigmoid(np.dot(w, X.T) + b)

        r = X.shape[0]
        cost = (-1 / r) * (np.sum((Y.T * np.log(result)) + ((1 - Y.T) * (np.log(1 - result)))))

        # Gradient calculation
        dw = (1 / r) * (np.dot(X.T, (result - Y.T).T))
        db = (1 / r) * (np.sum(result - Y.T))

        gradiants = {"dw": dw, "db": db}

        return gradiants, cost

    def fit(self,w, b, X, Y, epochs, lr):
        cost_array = []
        for e in range(epochs):
            gradians, cost = self.model(w, b, X, Y)

            dw = gradians["dw"]
            db = gradians["db"]

            # weight update
            w = w - (lr * (dw.T))
            b = b - (lr * db)

            if (e % 100 == 0):
                cost_array.append(cost)

        chkptw = w
        chkptb = b
        wb_optimize = {"w": chkptw, "b": chkptb}
        gradients = {"dw": dw, "db": db}

        return wb_optimize, gradients, cost_array

    def predict(self,z, rows):
        y_pred = np.zeros((1, rows))
        for i in range(z.shape[1]):
            if z[0][i] > 0.5:
                y_pred[0][i] = 1
        return y_pred
