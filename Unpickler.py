import pickle


class Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Perceptron':
            from Perceptron import Perceptron
            return Perceptron
        return super().find_class(module, name)
