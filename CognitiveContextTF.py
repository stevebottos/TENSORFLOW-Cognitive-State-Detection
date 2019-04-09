import tensorflow as tf
import numpy as np

# Functions

# To print outputs within Tensorflow session:
def tprint(x):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dprint(sess.run(x))

# Randomly initializes weights for NN
def initWeights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    bias = tf.random_normal([1, shape[1]], stddev=0.1)
    return tf.Variable(weights), tf.Variable(bias)

# Takes .csv data from a certain candidate, encodes it, and converts to Tensorflow's format
def getData(cand):
    apnd = str(cand)
    # Loading Training Data
    X_train = np.array(np.genfromtxt('C:\\Users\\HP\\PycharmProjects\\Tensorflow\\data\\trainset'+ apnd + '.csv', delimiter=","))
    y_train = np.array([ np.genfromtxt('C:\\Users\\HP\\PycharmProjects\\Tensorflow\\data\\ytrain'+ apnd + '.csv', delimiter=",") ]).T

    # Shuffling
    n = X_train.shape[1]
    t = np.concatenate((X_train, y_train), 1)
    shuff = np.random.permutation(t)
    X_train = shuff[:, 0:n]
    y_train = shuff[:, n]

    # Converting to a "one-hot" style vector:
    y = np.zeros((len(y_train), 3))
    y[:, 0] = y_train == 1
    y[:, 1] = y_train == 2
    y[:, 2] = y_train == 3
    y_train = y

    # Loading Test Data
    X_test = np.array(
        np.genfromtxt('C:\\Users\\HP\\PycharmProjects\\Tensorflow\\data\\testset'+ apnd + '.csv', delimiter=","))
    X_test = X_test[:, 0:2]
    y_test = np.array([ np.genfromtxt('C:\\Users\\HP\\PycharmProjects\\Tensorflow\\data\\ytest'+ apnd + '.csv', delimiter=",") ]).T

    # Shuffling
    n = X_test.shape[1]
    t = np.concatenate((X_test, y_test), 1)
    shuff = np.random.permutation(t)
    X_test = shuff[:, 0:n]
    y_test = shuff[:, n]

    # Converting to a "one-hot" style vector:
    y = np.zeros((len(y_test), 3))
    y[:, 0] = y_test == 1
    y[:, 1] = y_test == 2
    y[:, 2] = y_test == 3
    y_test = y

    return X_train, y_train, X_test, y_test

# The forward-propagation step for each layer
def forwardprop(X, w_1, b_1, w_2, b_2):
    h = (tf.add(tf.matmul(X, w_1), b_1))
    h = tf.nn.sigmoid(h)
    yhat = tf.add(tf.matmul(h, w_2), b_2)
    return yhat

# The main function, the TF session runs in here
def main(cand):

    # Retrieving data:
    X_train, y_train, X_test, y_test = getData(cand)

    # Layer's sizes:
    in_size = X_train.shape[1]  # = 6
    h_size = 2                  # Number of hidden nodes
    y_size = y_train.shape[1]   # = 3

    # Symbols
    m,n = X_train.shape
    X = tf.placeholder("float", shape=[None,n])
    m, n = y_train.shape
    y = tf.placeholder("float", shape=[None,n])

    # Weight initializations
    w_1, b_1 = initWeights((in_size, h_size))
    w_2, b_2 = initWeights((h_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1, b_1, w_2, b_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat))
    optimizer = tf.train.AdamOptimizer(0.02).minimize(cost)

    # Run SGD
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1000):
            sess.run(optimizer, feed_dict={X: X_train, y: y_train})

            test_accuracy = np.mean(np.argmax(y_test, axis=1) ==
                                    sess.run(predict, feed_dict={X: X_test, y: y_test}))

            print(np.round(test_accuracy * 100, 2))



        w1 = sess.run(w_1)
        b1 = sess.run(b_1)
        w2 = sess.run(w_2)
        b2 = sess.run(b_2)

        return w1, b1, w2, b2, test_accuracy

accuracy = []


for cand in range(1,41):
    w1, b1, w2, b2, test_accuracy = main(cand)
    test_accuracy = np.array([[test_accuracy]])

    if len(accuracy) == 0:
        accuracy = test_accuracy
        print(accuracy.shape)
    else:
        accuracy = np.concatenate((accuracy, test_accuracy))
        np.array([accuracy])

    print(np.round(accuracy*100, 2))