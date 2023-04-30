from abc import abstractclassmethod

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
        raise NotImplementedError
    
    @abstractclassmethod
    def input(self):
        return self.input

    @abstractclassmethod
    def output(self):
        return self.output
    
    @abstractclassmethod
    def input_shape(self):
        return self.input_shape
    
    @abstractclassmethod
    def output_shape(self):
        return self.output_shape
    
    @abstractclassmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractclassmethod
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
