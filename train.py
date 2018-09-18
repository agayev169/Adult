import time
import numpy as np
import random

train = []
test = []

MAX_AGE = 90.0
WORKCLASS = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']
MAX_FNLWGT = 1484705
EDUCATION = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', 
'7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?']
MAX_EDUCATION_NUM = 16.0
MARTIAL_STATUS = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?']
OCCUPATION = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?']
RELATIONSHIP = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?']
RACE = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?']
SEX = ['Female', 'Male', '?']
MAX_CAPITAL_GAIN = 9999.0
MAX_CAPITAL_LOSS = 4356.0
MAX_HOURS_PER_WEEK = 99.0
NATIVE_COUNTRY = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 
'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 
'Peru', 'Hong', 'Holand-Netherlands', '?']


DATA_NORMALIZER = [MAX_AGE, WORKCLASS, MAX_FNLWGT, EDUCATION, MAX_EDUCATION_NUM, MARTIAL_STATUS, OCCUPATION, RELATIONSHIP, RACE, SEX,
MAX_CAPITAL_GAIN, MAX_CAPITAL_LOSS, MAX_HOURS_PER_WEEK, NATIVE_COUNTRY]

def normalize(l):
	try:
		l_ = []
		for i in range(15):
			if i == 0 or i == 2 or i == 4 or i == 10 or i == 11 or i == 12:
				l_.append(float(l[i]) / DATA_NORMALIZER[i])
			elif i == 14:
				if '<' in l[i]:
					l_.append(0)
				else:
					l_.append(1)
			else:
				l_.append(DATA_NORMALIZER[i].index(l[i]) / len(DATA_NORMALIZER[i]))
	except:
		pass
	return l_

t = time.clock_gettime(0)
f = open("train", "r")
lines = f.readlines()
for l in lines:
	l_ = l.split(', ')
	train.append(normalize(l_))

X = []
y = []

for i in range(len(train)):
	if len(train[i]) != 15:
		continue
	X.append(train[i][:14])
	y.append([train[i][14]])

X = np.array(X)
y = np.array(y)


test = []

f = open("test", "r")
lines = f.readlines()
for l in lines:
	l_ = l.split(', ')
	test.append(normalize(l_))

X_val = []
y_val = []

for i in range(len(test)):
	if len(test[i]) != 15:
		continue
	X_val.append(test[i][:14])
	y_val.append([test[i][14]])

X_val = np.array(X_val)
y_val = np.array(y_val)

# X = X + X_val
# y = y + y_val

# X = np.array(X)
# y = np.array(y)

print("Time spent to load test data: " + str(time.clock_gettime(0) - t))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard

# n_layers = [1, 2, 3]
n_layers = [2]
# n_nodes = [256, 512, 1024]
n_nodes = [256]
# learning_rates = [0.01, 0.001, 0.0001]
learning_rates = [0.001]

for n_layer in n_layers:
	for n_node in n_nodes:
		for lr in learning_rates:

			# NAME = 'Adult-{}x{}-{}-{}'.format(n_node, n_layer, lr, int(time.time()))

			# print(NAME)

			# tb = TensorBoard(log_dir = 'logs/{}'.format(NAME))

			model = Sequential()
			model.add(Flatten())

			for _ in range(n_layer):
				model.add(Dense(n_node))
				model.add(Activation('tanh'))
				model.add(Dropout(0.1))

			model.add(Dense(1))
			model.add(Activation('sigmoid'))

			opt = optimizers.Adam(lr = lr)

			model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy'])

			model.fit(X, y, epochs = 30, batch_size = 32, validation_data = (X_val, y_val))
			# model.fit(X, y, epochs = 15, batch_size = 64, validation_split = 0.5, callbacks = [tb])
			# model.evaluate(X_val, y_val)	