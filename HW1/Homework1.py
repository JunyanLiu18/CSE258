import numpy
import scipy.optimize
import random
from matplotlib import pyplot as plt
from collections import defaultdict


import csv
import gzip

# from sklearn import linear_model


path = "/Users/apple/Desktop/CSE258/amazon_reviews_us_Gift_Card_v1_00.tsv.gz"
c = csv.reader(gzip.open(path, 'rt'), delimiter = '\t')
dataset = []
first = True
for line in c:
    # The first line is the header
    if first:
        header = line
        first = False
    else:
        d = dict(zip(header, line))
        # Convert strings to integers for some fields:
        d['star_rating'] = int(d['star_rating'])
        d['helpful_votes'] = int(d['helpful_votes'])
        d['total_votes'] = int(d['total_votes'])
        dataset.append(d)
# print(header)

# print(dataset[1])

count_1star = [d['star_rating'] == 1 for d in dataset]
count_2star = [d['star_rating'] == 2 for d in dataset]
count_3star = [d['star_rating'] == 3 for d in dataset]
count_4star = [d['star_rating'] == 4 for d in dataset]
count_5star = [d['star_rating'] == 5 for d in dataset]
# print(sum(count_1star), sum(count_2star), sum(count_3star), sum(count_4star), sum(count_5star))



# def feature(datum):
# 	feat = [1]
# 	if datum['verified_purchase'] == "Y":
# 		feat.append(1)
# 	elif datum['verified_purchase'] == "N":
# 		feat.append(0)
# 	feat.append(len(datum['review_body']))
# 	return feat

# x = [feature(d) for d in dataset]
# y = [d['star_rating'] for d in dataset]
# theata1, residuals, rank, s = numpy.linalg.lstsq(x, y)
# print(theata1)


def feature(datum):
	feat = [1]
	if datum['verified_purchase'] == "Y":
		feat.append(1)
	else:
		feat.append(0)
	return feat

x = [feature(d) for d in dataset]
y = [d['star_rating'] for d in dataset]
# theata2, residuals, rank, s = numpy.linalg.lstsq(x, y)
# print(theata2)



def split_training(x, a):
	return x[:int(len(x) * a)]
def split_testing(x, a):
	return x[int(len(x) * a):]


x = [feature(d) for d in dataset]
x_training = split_training(x, 0.9)
x_testing = split_testing(x, 0.9)
y = [d['star_rating'] for d in dataset]
y_training = split_training(y, 0.9)
y_testing = split_testing(y, 0.9)
theata3, residuals, rank, s = numpy.linalg.lstsq(x_training, y_training)
# print(theata3)






def mse_calc(x, y, theata):
	count = 0
	mse = 0
	while(count < len(x)):
		mse += (y[count] - numpy.dot(x[count],theata)) ** 2
		count = count + 1
	mse = mse/len(x)
	return mse
mse_training = mse_calc(x_training, y_training, theata3)
mse_testing = mse_calc(x_testing, y_testing, theata3)
print(mse_training, mse_testing)

prct = 0.05
mse_res = []
xplot = []
while (prct <= 0.95):
	xplot.append(prct)
	x_training = split_training(x, prct)
	x_testing = split_testing(x, prct)
	y_training = split_training(y, prct)
	y_testing = split_testing(y, prct)
	theata, residuals, rank, s = numpy.linalg.lstsq(x_training, y_training)
	mse_res.append([mse_calc(x_training, y_training, theata)])
	prct = prct + 0.01
print(mse_res)

plt.plot(xplot, mse_res)

# model = linear_model.LogisticRegression()
# model.fit(x_training, y_training)
# predictions = model.predict(x_training)
# print(predictions)

p = ["Y" in d['verified_purchase'] for d in dataset]
p = sum(p) * 1.0 / len(p)
