import re
import numpy as np
from numpy import random
import copy

np.seterr(all='ignore')
alpha = 0.1
eL = 1.0
Weightlist1 = []
Weightlist2 = []
sigL = -1.0

for i in range(96000):
    Weightlist1.append(random.uniform(-0.1, 0.1))
for i in range(100):
    Weightlist2.append(random.uniform(-0.1, 0.1))


def read_pgm(filename,  plot=False, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer1 = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer1).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer1,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def sigmoid(s):
    return 1 / (1 + np.exp(np.longfloat(-s)))


def Calculate_error(s, label):
    return label - s


def delta(e, s):
    return e * (s - s ** 2)


def layer(new_image, label):
    global alpha, eL, Weightlist1, Weightlist2, sigL
    deltaL = 0.0
    deltaL1 = np.zeros(100)
    image = np.array(new_image)
    Weightlist1 = np.array(Weightlist1)
    Weightlist2 = np.array(Weightlist2)
    output = np.zeros(100)
    count = 0
    while count < 1000:
        count += 1
        temp_output = 0.0
        k = 0
        for i in range(0, 100):
            S = 0.0
            for j in range(0, 960):
                S += image[j] * Weightlist1[k]
                k += 1
            temp_sigmoid = sigmoid(S)
            output[i] = temp_sigmoid
            temp_output += temp_sigmoid * Weightlist2[i]
        sigL = sigmoid(temp_output)
        eL = Calculate_error(sigL, label)
        if eL >= -0.04 and eL <= 0.04:
            break
        error = copy.deepcopy(eL)
        deltaL = delta(error, sigL)
        output = np.array(output)
        for i in range(0, 100):
            Weightlist2[i] += (output[i] * deltaL * alpha)
        u = 0.0
        for i in range(0, 100):
            u += Weightlist2[i] * deltaL
        for j in range(0,100):
            n = output[j] * (1 - output[j])
            for i in range(0,960):
                de = j + image[i] * n * u
            deltaL1[j] = de
        for d in range(0, 100):
            for i in range(0, 960):
                Weightlist1[i] += alpha * image[i] * deltaL1[d]
    return sigL


def predict(new_image):
    global Weightlist1, Weightlist2
    eL = 2.0
    image = np.array(new_image)
    k = 0
    temp_output = 0.0
    p = 0.0
    output = np.zeros(100)
    output = np.array(output)
    for i in range(0, 100):
        S = 0.0
        for j in range(0, 960):
            S += image[j] * Weightlist1[k]
            k = k + 1
        temp_sigmoid = sigmoid(S)
        output[i] = temp_sigmoid
        temp_output += temp_sigmoid * Weightlist2[i]
    sigL = sigmoid(temp_output)
    return sigL


train_file = 'downgesture_train.list'
test_file = 'downgesture_test.list'
trainfile = open(train_file,'r')

for training_image in trainfile:
    training_image = training_image[:-1]
    image = read_pgm(training_image)
    new_image = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > 0:
                new_image.append(float(image[i][j]))
            else:
                new_image.append(0)
            if 'down' in training_image:
                label = 1
            else:
                label = 0
    sig = layer(new_image, label)
    if sig < 0.0001:
        sig = 0.0
    print "Train: Training of {} is with value: {}".format(training_image, sig)

print " \n********************   test cases   ********************\n"

trainfile.close()
testfile = open(test_file,'r')

total = 0.0
correct = 0.0

for test_image in testfile:
    total += 1
    test_image = test_image[:-1]
    image = read_pgm(test_image)
    new_image = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > 0:
                new_image.append(float(image[i][j]))
            else:
                new_image.append(0)
    beta = predict(new_image)
    if beta <= 0.2:
        beta = 0
    if beta > 0.2:
        beta = 1
    print "Test: Prediction of {} is {}".format(test_image, beta)
    if 'down' in test_image:
        if beta == 1:
            correct += 1
    else:
        if beta == 0:
            correct += 1

print '\nAccuracy: {}'.format(correct * 100 / total)
