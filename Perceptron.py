import numpy


class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = numpy.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        print(self.w_[1:], self.w_[0])
        return numpy.dot(X, self.w_[1:]) + self.w_[0]

    # Function change
    def predict(self, X):
        if numpy.where(self.net_input(X) >= 0, 1, -1) > 0:
            return "Versicolor"
        else:
            return "Setosa"
