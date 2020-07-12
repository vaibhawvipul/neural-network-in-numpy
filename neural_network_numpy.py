import numpy as np
import math 

np.random.seed(0)

class NeuralNet:

    def __init__(self, layers=[1,100,1], activations=['relu','relu']):
        assert(len(layers) == len(activations)+1)
        self.layers = layers 
        self.activations = activations

        self.weights = []
        self.biases = []

        #initialize weights
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(self.layers[i+1], self.layers[i]))
            self.biases.append(np.random.randn(self.layers[i+1], 1))

    @staticmethod
    def getActivationFunction(name):
        if(name == 'sigmoid'):
            return lambda x : np.exp(x)/(1+np.exp(x))
        else:
            print("Unknown. Using Linear instead")
            return lambda x: x 

    def feedforward(self, x):

        # feedforward loop starts
        input_to_layer = np.copy(x) # why use np.copy?

        output_from_layers = []
        intermediate_inputs = [input_to_layer]

        for j in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[j])
            output_from_layers.append(self.weights[j].dot(input_to_layer)+self.biases[j])
            input_to_layer = activation_function(output_from_layers[-1])
            intermediate_inputs.append(input_to_layer)

        return (output_from_layers, intermediate_inputs)

    @staticmethod
    def getDerivitiveActivationFunction(name):
        if(name == 'sigmoid'):
            sig = lambda x : np.exp(x)/(1+np.exp(x))
            return lambda x :sig(x)*(1-sig(x))
        else:
            print('Unknown activation. Using Linear instead')
            return lambda x: 1

    def backpropagation(self, y, output_from_layers, intermediate_inputs):
        # Notes:  elementwise multiplication is sometimes called 
        # the Hadamard product or Schur product
        dw = [] #dC/dW
        db = [] #dC/dW
        deltas = [None] * len(self.weights) #dC/dZ errors for each layer

        # insert the last layer error
        deltas[-1] = ((y-intermediate_inputs[-1])*self.getDerivitiveActivationFunction(self.activations[-1])(output_from_layers[-1]))

        # 4 equations of backpropagation and their proofs.
        for i in reversed(range(len(deltas)-1)):
            # self.getDerivitiveActivationFunction(self.activations[i] will return a function, we need to pass an x to it.
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(output_from_layers[i])) 

            batch_size = y.shape[1]
            db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]

            dw = [d.dot(intermediate_inputs[i].T)/float(batch_size) for i,d in enumerate(deltas)]

        return dw, db

    def train(self, x,y,batch_size=10, epochs=10, lr=0.01):
        for e in range(epochs):
            i = 0
            while i<len(y):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                i = i+batch_size

                output_from_layers, intermediate_inputs = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, output_from_layers, intermediate_inputs)

                self.weights = [w+lr*dweight for w,dweight in zip(self.weights, dw)]
                self.biases = [w+lr*dbias for w,dbias in zip(self.biases, db)]

                print("loss = {}".format(np.linalg.norm(intermediate_inputs[-1]-y_batch)))

    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    nn = NeuralNet([1, 100, 1],activations=['sigmoid', 'sigmoid'])
    X = 2*np.pi*np.random.rand(1000).reshape(1, -1)
    y = np.sin(X)
    
    nn.train(X, y, epochs=1000, batch_size=64, lr = .1)
    _, a_s = nn.feedforward(X)
    #print(y, X)
    plt.scatter(X.flatten(), y.flatten())
    plt.scatter(X.flatten(), a_s[-1].flatten())
    plt.show()
