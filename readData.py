# Get as input the filename containing Document	Category data
# Return proxies for Documents -> Int, Category -> Int
# Return the data as a list containing [Doc_proxy, Cat_proxy] pairs
#
from collections import defaultdict
import sys
from sys import stdout

def read(filename):
	data_points_read = 0
	docs_read = 0
	cats_read = 0
	fi = open(filename, "r")
	docs = defaultdict()
	cats = defaultdict()
	data = []
	for line in fi:
		a = line.split("\t", 1)
		if(len(a) > 1):
			doc = a[0].strip()
			if doc not in docs.keys():
				docs[doc] = docs_read
				docs_read = docs_read + 1
			categories = a[1].split("\t")
			for c in categories:
				if c not in cats.keys():
					cats[c] = cats_read
					cats_read = cats_read + 1
				data_points_read = data_points_read + 1
				data.append([docs[doc], cats[c]])

		if(docs_read % 1000 == 0):
			#print docs_read
			sys.stdout.write(str(docs_read) + ", ")
	print "\nDocs : ", docs_read, "Categories : ", cats_read, "Data Points : ", len(data) 
	return docs, cats, data		


def readYelp(filename):
	data_points_read = 0
	buss_read = 0
	atts_read = 0
	fi = open(filename, "r")
	buss = defaultdict()
	atts = defaultdict()
	data = []
	bus_id = ""
	att = ""
	for line in fi:
		a = line.split(":")
		if(len(a) >= 2):
			if(a[0].strip() == "business_id"):
				bus_id = a[1].strip();
				#buss[bus_id] = buss_read	
				#buss_read = buss_read + 1
			else:
				att = a[0].strip()
				if att not in atts.keys():
					atts[att] = atts_read
					atts_read = atts_read + 1

				if bus_id not in buss.keys():
					buss[bus_id] = buss_read
					buss_read = buss_read + 1	
				
				truth = int(a[1].strip())
				data.append( [buss[bus_id], atts[att], truth] )
				data_points_read = data_points_read + 1
	
	return buss, atts, data		

def readAmazon(filename):
	data_points_read = 0
	buss_read = 0
	atts_read = 0
	fi = open(filename, "r")
	buss = defaultdict()
	atts = defaultdict()
	data = []
	for line in fi:
		a = line.split(" ", 3)
		if(len(a) >= 2):
			bus = a[0].strip()
			user = a[1].strip()
			rate = float(a[2].strip())
			if (rate >= 4):
				rate = 1
			else:
				rate = 0

			if bus not in buss.keys():
				buss[bus] = buss_read
				buss_read = buss_read + 1	
			if user not in atts.keys():
				atts[user] = atts_read
				atts_read = atts_read + 1			


			data.append( [buss[bus], atts[user], rate] )
			data_points_read = data_points_read + 1
			if(data_points_read % 1000 == 0):
				print data_points_read

	return buss, atts, data			




	

if __name__=="__main__":
	filename = sys.argv[1]
	if(sys.argv[2] == "yelp"):
		print "ok"
		buss, atts, data = readAmazon(filename);
		print len(buss.keys()), len(atts.keys())
		print data[27000:27900]
		
				




