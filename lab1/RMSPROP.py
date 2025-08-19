import numpy

import SGD

class RMSPropGradientDescent(SGD.StochasticGradientDescent):
	def __init__(self, epochs_number, batch_size, learning_rate=0.01, eps=1e-3, a: float =0.99, m: float=0):
		SGD.StochasticGradientDescent.__init__(self, epochs_number, batch_size, learning_rate, eps)
		self.a=a
		self.m=m
		self.v_weights=0
		self.b_weights=0
		self.v_bias=0
		self.b_bias=0

	def makeStep(self, weights_gradient, bias_gradient):
		self.v_weights=self.a*self.v_weights + (1-self.a)*(weights_gradient**2)
		self.v_bias=self.a*self.v_bias + (1-self.a)*(bias_gradient**2)
		if self.m>0:
			self.b_weights=self.m*self.b_weights + weights_gradient/(numpy.sqrt(self.v_weights)+self.eps)
			self.b_bias=self.m*self.b_bias + bias_gradient/(numpy.sqrt(self.v_bias)+self.eps)
			self.weights-=self.learning_rate*self.b_weights
			self.bias-=self.learning_rate*self.b_bias
		else:
			self.weights-=self.learning_rate*weights_gradient/(numpy.sqrt(self.v_weights)+self.eps)
			self.bias-=self.learning_rate*bias_gradient/(numpy.sqrt(self.v_bias)+self.eps)
			