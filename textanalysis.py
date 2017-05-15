import nltk
import csv
from nltk.probability import FreqDist
import numpy as np
from nltk.corpus import stopwords


####### NEURAL NETWORK #################


class NeuralNet:
    def __init__(self):
        self.W_0 = np.random.uniform(-0.00048828125, 0.00048828125, size=(15, 1182)) #weights from hidden layer <- V input nodes
        self.W_1 = np.random.uniform(-0.00048828125, 0.00048828125, size=(4, 15)) #weights from output layer <- hidden layer nodes

        self.B_0 = np.random.randn(15, 1)/100
        self.B_1 = np.random.randn(4, 1)/100

        self.V = [] # keeps track of the inputs
        self.O_hidden = [] #keeps track of outputs from the hidden layer
        self.O_output = [] #keeps track of outputs from the output layer

        self.D_1 = [] #error vector

        self.learning_rate = 0.01

        self.error = []

    def tanh_derivative(self, o):
        return 1 - o**2

    def forward(self, v, actual):
        self.V.append(v)

        o_hidden = np.tanh(np.add(np.dot(self.W_0, v), self.B_0))
        self.O_hidden.append(o_hidden)

        o_output = np.tanh(np.add(np.dot(self.W_1, o_hidden), self.B_1))
        self.O_output.append(o_output)

        e = actual - o_output
        self.error.append(e)
        derivFunc = np.vectorize(self.tanh_derivative)
        o_deriv = derivFunc(o_output)
        e_d = e * o_deriv

        self.D_1.append(e_d)

        return o_output

    def normal_forward(self, v):
        """
        Assumes v is a row dominant input vector that
        needs to be transposed
        """
        v = np.transpose([v])

        o_hidden = np.tanh(np.dot(self.W_0, v) + self.B_0)
        o_output = np.tanh(np.dot(self.W_1, o_hidden) + self.B_1)

        return o_output

    def backward(self):
        d_1 = self.D_1[0]
        self.W_1 += np.dot(d_1, np.transpose(self.O_hidden[0])) * self.learning_rate
        self.B_1 += d_1 * (np.absolute(self.B_1) / self.B_1) * self.learning_rate

        d_0 = np.dot(np.transpose(self.W_1), d_1)
        derivFunc = np.vectorize(self.tanh_derivative)
        o_deriv = derivFunc(self.O_hidden[0])
        d_0 = o_deriv * d_0

        self.W_0 += np.dot(d_0, np.transpose(self.V[0])) * self.learning_rate
        self.B_0 += d_0 * (np.absolute(self.B_0) / self.B_0) * self.learning_rate

    def epoch(self, v, expected):
        """
        Performs one iteration of training

        V -> input (row dominant, normal array)
        E -> Expected (row dominant, normal array)
        """
        self.V = []
        self.O_hidden = []
        self.O_output = []
        self.D_1 = []

        self.error = []


        self.forward(np.transpose([v]), np.transpose([expected]))
        self.backward()

    def calculate_error(self):
        return sum(self.error[0])/len(self.error[0])



n = NeuralNet()


#for i in range(100):
#    input_vector = np.zeros(1284)
#    input_vector[1] = 1
#    expected_vector = np.zeros(4)
#    expected_vector[0] = 0.3
#    expected_vector[2] = 0.7
#    n.epoch(input_vector, expected_vector)
#    print n.calculate_error()
#
#input_vector = np.zeros(1284)
#input_vector[1] = 1
#print n.normal_forward(input_vector)


#sentiment ranges from 0 onwards
# for example: 0 means bad and 4 means a positive sentiment


f = open("/Users/samcarbet/Desktop/grade-12-assignments-wenqinYe/04e - Assignment - Sorted Visualization/smith.txt", "r")

txt = f.read()
words = nltk.tokenize.word_tokenize(txt)
fdist = FreqDist(words)
words = fdist.keys()

word_dict = {}

#conevert words to dict
index = 0
for word in words:
    word_dict[word] = index
    index += 1

s = set(stopwords.words('english'))
def sent2vec(sentence, word_dict):
    """
    """
    vec = np.zeros(1182)
    words = filter(lambda w: not w in s, nltk.word_tokenize(sentence))
    for word in words:
        if word in word_dict:
            vec[word_dict[word]] = 1

    return vec

def calculate_text(text):
    return n.normal_forward(sent2vec(text, word_dict))

#for epoch in range(3):
#    with open("/Users/samcarbet/Downloads/train.tsv") as tsvfile:
#        tsvreader = csv.reader(tsvfile, delimiter="\t")
#        current_line = 0
#        limit = 150000
#
#        for line in tsvreader:
#
#            current_line+= 1
#            if current_line < 2:
#                continue
#            if current_line > limit:
#                break
#
#            if current_line % 1000 == 0:
#                print "loop: {}".format(current_line)
#                sent_vec = sent2vec("By this time, Dobyns' conscience was hurting him badly", word_dict)
#                print n.normal_forward(sent_vec)
#
#            vec = sent2vec(line[2], word_dict)
#            sentiment = line[3]
#            sent_vec = np.full((4), -1)
#            sent_vec[int(sentiment)-1] = 1
#
#            error = float("inf")
#            iter_count = 0
#            while error > 0.01 and iter_count < 100:
#                n.epoch(vec, sent_vec)
#                error = n.calculate_error()
#
#                iter_count += 1
#

#sent_vec = sent2vec("hello this sucks bad", word_dict)
#print n.normal_forward(sent_vec)
