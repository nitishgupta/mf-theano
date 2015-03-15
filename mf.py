from collections import defaultdict
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np

def readData(filename):
	fi = open(filename, "r")
	entities = defaultdict()
	data = defaultdict()
	for line in fi:
		a = line.split()
		if(len(a) > 1):
			e1 = a[0].strip();
			e2 = a[1].strip();
			rate = float(a[2].strip())
			if e1 not in entities.keys():
				entities[e1] = len(entities) + 1
			if e2 not in entities.keys():
				entities[e2] = len(entities) + 1
			
			if entities[e1] not in data.keys():
				data[entities[e1]] = defaultdict()
			data[entities[e1]][entities[e2]] = rate
	return	entities, data

def printData(data):
	for key in data:
		print key
		for key1 in data[key]:
			print "\t", key1, data[key][key1]	


def mf(entities, data, K, lr):
	srng = RandomStreams(seed=234)
	rng = np.random
	rate_t = T.fscalar('rate_true')
	e1 = T.iscalar('e1')
	e2 = T.iscalar('e2')
	alpha = theano.shared(rng.normal(loc=0.0, scale=0.01), name="alpha") 
	b = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(entities))), name="bias")
	phi = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(entities), K)), name="phi")
	
	b1, b2, phi1, phi2 = b[e1], b[e2], phi[e1], phi[e2]
	prediction = alpha + b1 + b2 + T.dot(phi1, phi2)
	cost = (prediction - rate_t)*(prediction - rate_t)

	#ga, gb1, gb2, gphi1, gphi2 = T.grad(cost, [alpha, b1, b2, phi1, phi2])
	ga, gb, gphi = T.grad(cost, [alpha, b, phi])
	
	predict = theano.function(inputs=[e1, e2], outputs = prediction)
	c = theano.function(inputs=[e1, e2, rate_t], outputs=cost)
	train = theano.function(inputs=[e1, e2, rate_t], outputs=[prediction, cost], updates= ( (b, b - lr*gb), (alpha, alpha - lr*ga), (phi, phi - lr*gphi)) )


	print predict(1,2), c(1,2,4.0)

	print phi.get_value()[1]
	print phi.get_value()[1]  - (predict(1,2) - 4.0)*phi.get_value()[2]*2*0.01
	train(1,2,4.0);
	print phi.get_value()[1]


if __name__ == "__main__":
	entities, data = readData("data.txt")		
	print entities
	print data
	mf(entities, data, 5, 0.01)
			
