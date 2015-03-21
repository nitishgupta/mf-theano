from collections import defaultdict
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf
from random import shuffle
import sys
import random as rng
import readData


# def mf(trdata, vdata, tdata, K, lr, reg, epoch):
# 	srng = RandomStreams(seed=234)
# 	rng = np.random
# 	truth = T.dscalar('rate_true')
# 	reg_con = T.dscalar('lambda')
# 	e1 = T.iscalar('e1')
# 	e2 = T.iscalar('e2')
# 	alpha = theano.shared(rng.normal(loc=0.0, scale=0.0001), name="alpha") 
# 	b_docs = theano.shared(rng.normal(loc=0.0, scale=0.0001, size=(len(docs.keys()))), name="bias_docs")
# 	b_cats = theano.shared(rng.normal(loc=0.0, scale=0.0001, size=(len(cats.keys()))), name="bias_cats")
# 	phi_e1 = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(docs.keys()), K)), name="phi_e1")
# 	phi_e2 = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(cats.keys()), K)), name="phi_e2")

# 	prob = 1 / (1 + T.exp(- (alpha + b_docs[e1] + b_cats[e2] + T.dot(phi_e1[e1], phi_e2[e2])) ))
# 	prediction = prob > 0.5
# 	cost = truth*T.log(prob) + (1 - truth)*(1 - T.log(prob)) - reg_con*(alpha*alpha + b_docs[e1]*b_docs[e1] + b_cats[e2]*b_cats[e2] + T.dot(phi_e1[e1],phi_e1[e1]) + T.dot(phi_e2[e2],phi_e2[e2]) )

# 	#ga, gb1, gb2, gphi1, gphi2 = T.grad(cost, [alpha, b1, b2, phi1, phi2])
# 	ga, gbd, gbc, gphid, gphic = T.grad(cost, [alpha, b_docs, b_cats, phi_e1, phi_e2])

# 	predict = theano.function(inputs=[e1, e2], outputs = prediction)
# 	c = theano.function(inputs=[e1, e2, truth, theano.Param(reg_con, default=reg)], outputs=cost)
# 	train = theano.function(inputs=[e1, e2, truth, theano.Param(reg_con, default=reg)], outputs=[prediction, cost], updates= ( (b_docs, b_docs + lr*gbd), (b_cats, b_cats + lr*gbc), (alpha, alpha + lr*ga),
# 																																(phi_e1, phi_e1 + lr*gphid), (phi_e2, phi_e2 + lr*gphic)) )

# 	mae_val = 0.0
# 	mae_test = 0.0
# 	#print predict(1192, 1820);
# 	for i in range(0, epoch):
# 		shuffle(trdata);
# 		for d in trdata:
# 			train(d[0], d[1], 1)		## Positive Sample e1, e2 
# 			neg_cat = rng.randint(0, len(cats.keys()) - 1)
# 			train(d[0], neg_cat, 0)

# 		for d in vdata:
# 			print predict(d[0], d[1]), 

# 		# mae_val = mae_val / len(vdata)
# 		# for d in tdata:
# 		# 	mae_test = mae_test + pow(predict(d[0], d[1]) - d[2], 2)
# 		# mae_test = mae_test / len(tdata)

# 		# print "Epoch : ", i, "validation mean sq. error : ", mae_val
# 		# print "Epoch : ", i, "testdataa  mean sq. error : ", mae_test
# 		# print 



def mfYelp(trdata, vdata, tdata, K, lr, reg, epoch):
	srng = RandomStreams(seed=234)
	rng = np.random
	truth = T.dscalar('rate_true')
	reg_con = T.dscalar('lambda')
	e1 = T.iscalar('e1')
	e2 = T.iscalar('e2')
	phi_e1 = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(e1s.keys()), K)), name="phi_e1")
	phi_e2 = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(e2s.keys()), K)), name="phi_e2")

	#sigmoid = 
	#sigmoid = 1 / (1 + T.exp(- (alpha + b_docs[e1] + b_cats[e2] + T.dot(phi_e1[e1], phi_e2[e2]) ) ))
	pre_sigmoid_activation = T.dot(phi_e1[e1], phi_e2[e2])
	sigmoid = T.nnet.sigmoid(pre_sigmoid_activation)
	#sigmoid = 1 / (1 + T.exp(-T.dot(phi_e1[e1], phi_e2[e2])) )

	prediction = T.nnet.sigmoid(pre_sigmoid_activation) > 0.5
	#cost = truth*T.log(sigmoid) + (1 - truth)*(1 - T.log(sigmoid)) - reg_con*(0)
	cost = truth*T.log(T.nnet.sigmoid(pre_sigmoid_activation)) + (1 - truth)*(1 - T.log(T.nnet.sigmoid(pre_sigmoid_activation))) - reg_con*(T.dot(phi_e1[e1],phi_e1[e1]) + T.dot(phi_e2[e2],phi_e2[e2]) )

	#ga, gb1, gb2, gphi1, gphi2 = T.grad(cost, [alpha, b1, b2, phi1, phi2])
	gphi1, gphi2 = T.grad(cost, [phi_e1, phi_e2])

	prob = theano.function(inputs=[e1, e2], outputs = sigmoid)
	predict = theano.function(inputs=[e1, e2], outputs = prediction)
	c = theano.function(inputs=[e1, e2, truth, theano.Param(reg_con, default=reg)], outputs=cost)
	train = theano.function(inputs=[e1, e2, truth, theano.Param(reg_con, default=reg)], outputs=[prediction, cost], updates= ((phi_e1, phi_e1 + lr*gphi1), (phi_e2, phi_e2 + lr*gphi2)) )

	mae_val = 0.0
	mae_test = 0.0
	#print predict(1192, 1820);
	val_prediction = []
	val_true = []

	# print phi_e1.get_value()[10]
	# print phi_e2.get_value()[15]
	# print prob(10, 15), predict(10, 15)
	# print c(10, 15, 1)
	# train(10, 15, 1)
	# print phi_e1.get_value()[10]
	# print phi_e2.get_value()[15]
	# print prob(10, 15), predict(10, 15)
	# print c(10, 15, 1)
	# train(10, 15, 0)
	# print phi_e1.get_value()[10]
	# print phi_e2.get_value()[15]
	# print prob(10, 15), predict(10, 15)
	# print c(10, 15, 1)


	print "Initially "
	for d in vdata:
		val_prediction.append(predict(d[0], d[1]))
	countOnes(val_prediction)	
		

	for i in range(0, epoch):
		shuffle(trdata)
		train_cost = 0
		ones = 0
		total = 0
		for d in trdata:
			train(d[0], d[1], d[2])		## Positive Sample e1, e2
			train_cost = train_cost + c(d[0], d[1], d[2])
			if(d[2] == 1):
				ones = ones + 1
			total = total + 1	

		val_prediction = []			
		val_true = []
		for d in vdata:
			val_prediction.append(predict(d[0], d[1]))
			val_true.append(d[2])

		PRF = prf(val_true, val_prediction, average = 'micro');	
		p = round(PRF[0]*100, 1);
		r = round(PRF[1]*100, 1);
		f = round(PRF[2]*100, 1);
		print "Epoch : ", i, "P : ", p, " R : ", r, " F1 : ", f

	test_prediction = []			
	test_true = []
	for d in testdata:
		test_prediction.append(predict(d[0], d[1]))
		test_true.append(d[2])

	PRF = prf(test_true, test_prediction, average = 'micro');	
	p = round(PRF[0]*100, 1);
	r = round(PRF[1]*100, 1);
	f = round(PRF[2]*100, 1);
	print "Epoch : ", i, "P : ", p, " R : ", r, " F1 : ", f			
		


def countOnes(lis):
	ones = 0
	zeros = 0;
	for l in lis:
		if (l == 1):
			ones = ones + 1
		else:
			zeros = zeros + 1	

	print "zeros : ", zeros, " ones", ones		

def split_TrainValTest(trainperc, valperc):
	shuffle(data);
	trlen = int(len(data)*trainperc)
	vallen = int(len(data)*valperc)
	trdata = data[0:trlen]
	vdata = data[trlen:trlen+vallen]
	testdata = data[trlen+vallen:-1]
	return trdata, vdata, testdata

def getDataStats(dt):
	trues = 0
	false = 0
	for d in dt:
		if(d[2] == 1):
			trues = trues + 1
		else:
			false = false + 1

	print "true : ", trues, " false : ", false			


if __name__=="__main__":
	datafilename = sys.argv[1]
	#docs, cats, data = readDoc_Cat_Data.read(datafilename);
	e1s, e2s, data = readData.readYelp(datafilename);
	#docs, cats, data = readDoc_Cat_Data.readAmazon(datafilename);

	trdata, vdata, testdata = split_TrainValTest(0.8, 0.1);
	print len(data), len(trdata), len(vdata), len(testdata)
	print "Businesses : ", len(e1s.keys())
	getDataStats(trdata)
	getDataStats(vdata)
	getDataStats(testdata)
	

	#mf(trdata, vdata, testdata, 5, 0.01, 0.01, 20)
	mfYelp(trdata, vdata, testdata, K=30, lr=0.01, reg=0.001, epoch=100)