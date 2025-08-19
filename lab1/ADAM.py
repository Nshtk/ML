import numpy

import SGD

class AdaptiveGradientDescent(SGD.StochasticGradientDescent):
	def __init__(self, epochs_number, batch_size, learning_rate=0.01, eps=1e-3, b_1: float =0.9, b_2: float=0.99):
		SGD.StochasticGradientDescent.__init__(self, epochs_number, batch_size, learning_rate, eps)
		self.b_1=b_1
		self.b_2=b_2
		self.m_weights=0
		self.v_weights=0
		self.m_bias=0
		self.v_bias=0

	def makeStep(self, weights_gradient, bias_gradient):
		self.m_weights = self.m_weights*self.b_1 + (1-self.b_1) * weights_gradient
		self.v_weight = self.m_weights*self.b_1 + (1-self.b_1) * weights_gradient ** 2
		self.weights -= self.learning_rate * self.m_weights/(numpy.sqrt(self.v_weight)+self.eps)
		self.m_bias = self.m_bias*self.b_1 + (1-self.b_1) * bias_gradient
		self.v_bias = self.m_bias*self.b_1 + (1-self.b_1) * bias_gradient
		self.bias -= self.learning_rate * self.m_bias/(numpy.sqrt(self.v_bias)+self.eps)
		