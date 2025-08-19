import numpy as np

class StochasticGradientDescent():
	def __init__(self, epochs_number: int, batch_size: int, learning_rate: float=0.01, eps: float =1e-3):
		self.weights: float = None
		self.bias: float = None
		self.epochs_number = epochs_number
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.eps = eps
		
	def getGradient(self, x, y):
		error = np.dot(x, self.weights) + self.bias - y
		return np.dot(x.T, error) / x.shape[0], np.mean(error)
	def makeStep(self, weights_gradient, bias_gradient):
		self.weights -= self.learning_rate * weights_gradient
		self.bias -= self.learning_rate * bias_gradient
	def fit(self, x, y):
		samples_number, features_number = x.shape
		self.weights = np.random.randn(features_number)
		self.bias = np.random.randn()
	
		for i in range(self.epochs_number):
			indices_shuffled = np.random.permutation(samples_number)
			x_epoch = x[indices_shuffled]
			y_epoch = y[indices_shuffled]

			for j in range(0, samples_number, self.batch_size):
				weights_gradient, bias_gradient = self.getGradient(x_epoch[j:j+self.batch_size], y_epoch[j:j+self.batch_size])
				self.makeStep(weights_gradient, bias_gradient)

			if np.linalg.norm(weights_gradient) < self.eps:
				break
		return self.weights, self.bias
	
