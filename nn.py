import math,numpy as np

"""
activation
    Layer i
    [ 1 z0 z1 z2 ... zn ]

theta
    Layer i to i+1
    [ b    b    b    b    ...  b
      w0-0 w0-1 w0-2 w0-3 ... w0-m
      w1-0 w1-1 w0-2 w1-3 ... w1-m
      w2-0 w2-1 w0-2 w2-3 ... w2-m
      ...  ...  ...  ...  ...  ...
      ...  ...  ...  ...  ...  ...
      wn-0 wn-1 wn-2 wn-3 ... wn-m ]

"""
class Neural(object):

    def __init__(self, ar, eta, lamda):
        self.layers = len(ar)
        self.layerSizes = ar
        self.eta = eta
        self.lamda = lamda

        self.theta = {}
        self.thetaFlow = []
        self.activationFlow = []

        self.D = {}
        self.J = []
        self.currentEstimate = []
        self.JTest = []
        self.activation = {}
        self.activationPrime = {}
        self.theta[0] = np.array([[np.random.randint(1,100)/100 for _ in range(ar[1])] for _ in range(ar[0])])
        for i in range(1,len(ar)-1):
            self.theta[i] = np.array([[np.random.randint(1,100)/100 for _ in range(ar[i+1])] for _ in range(ar[i]+1)])

    def feed(self, X, Y):
        self.datasets = len(X)
        self.X = np.array(X).reshape(self.datasets,-1)
        self.Y = np.array(Y).reshape(self.datasets,-1)

        self.X = self.X /np.amax(self.X)
        self.Y = self.Y /np.amax(self.Y)

    def sigmoid(self,S):
        return 1/(1+np.exp(-S))

    def activationFuction(self,S):
        return self.sigmoid(S)

    def activationPrimeFunction(self,Z):
        return Z*(1-Z)

    def initActivation(self):
        for i in range(self.layers - 1):
            self.activation[i] = np.array([0 for _ in range(self.layerSizes[i]+1)])

        self.activation[self.layers-1] = np.array([0 for _ in range(self.layerSizes[-1])])

    def forward(self, dataset):
        self.activation[0] = self.X[dataset]

        for i in range(1, len(self.activation) - 1):
            # [1] + [activation i-1 * theta i-1]
            self.activation[i] = np.append(np.array([1]), self.activationFuction(self.activation[i-1].dot(self.theta[i-1])))

            self.activationPrime[i] = self.activationPrimeFunction(self.activation[i][1:]).T

        self.activation[self.layers-1] = self.activationFuction(self.activation[self.layers-2].dot(self.theta[self.layers-2]))
        self.activationPrime[self.layers-1] = self.activationPrimeFunction(self.activation[self.layers-1]).T

        self.currentEstimate.append(self.activation[self.layers-1])

    def backprop(self,dataset):
        yhat = self.Y[dataset]
        self.D[self.layers - 1] = (yhat - self.activation[self.layers - 1]).T


        for i in range(self.layers - 2, 0, -1):
            # We do not calculate deltas for the bias values
            theta_nobias = self.theta[i][0:-1, :]

            # delta[i] = (theta[i] . delta[i+1]) * activationPrime[i]
            self.D[i] = theta_nobias.dot(self.D[i+1]) * self.activationPrime[i]

        for i in range(0, self.layers-1):
            W_grad = -self.eta * (self.activation[i].reshape(-1,1).dot(self.D[i+1].reshape(1,-1)))
            self.theta[i] -= W_grad + self.eta * self.lamda * self.theta[i]

    def learn(self):
        self.initActivation()

        for i in range(len(self.X)):
            self.forward(i)
            self.backprop(i)
        self.J.append(self.loss())
        #self.test()
        #self.JTest.append(self.loss())

    def loss(self):
        weightSquareSum = 0
        for key in self.theta.keys():
            weightSquareSum += np.sum(self.theta[key]*self.theta[key])
        j = (self.lamda / 2) * weightSquareSum
        for i in range(len(self.currentEstimate)):
            j += self.cost(self.Y[i], self.currentEstimate[i])

        self.currentEstimate = []
        # stuff to examine the flow -----------------------------------------------------------------------------------
        # if the nn is large comment the following
        """
        self.thetaFlow.append(np.array([]))
        for key in self.theta.keys():
            self.thetaFlow[-1] = np.append(self.thetaFlow[-1], self.theta[key].reshape(1, -1))
        self.activationFlow.append(np.array([]))
        for key in self.activation.keys():
            self.activationFlow[-1] = np.append(self.activationFlow[-1], self.activation[key].reshape(1, -1))
        """
        #  --------------------------------------------------------------------------------------------------------------

        return j

    def test(self):
        for i in range(int(self.datasets*0.3)):
            self.forward(i)


    def cost(self, y_i, estimate_i):
        cost = 0
        for j in range(len(estimate_i)):
            estimate_i[j] = round(estimate_i[j])
            cost += (y_i[j] - estimate_i[j]) ** 2
        return cost * 0.5
        cost = 0
        for j in range(len(estimate_i)):

            if abs(estimate_i[j]) < 1e-320:
                a = -1000
            else:
                a = np.log(estimate_i[j])
            if abs(estimate_i[j] - 1) < 1e-320:
                b = -1000
            else:
                b = np.log(1 - estimate_i[j])
            cost -= y_i[j] * a + (1 - y_i[j]) * b
        return cost

def createData(equation, variables, size, scatter):
    x_data = []
    y_data = []
    for i in range(size):
        temp = equation
        x_data.append([])
        for j in range(len(variables)):
            x_data[-1].append(np.random.randint(1, 200)/100)
            temp = temp.replace(variables[j], str(x_data[-1][-1]))
        y_data.append(eval(temp))
    x_data = np.array(x_data)
    y_data = 1.*np.array(y_data)
    y_data += np.random.normal(size=y_data.shape, scale=((sum(y_data*y_data)**0.5)/size)*scatter/100)
    return x_data, y_data


def createDataClassification(features, classes, size, scatter):
    '''
    note that the range of a feature is 0 to 1
    :param features: number of features to explain a point
    :param classes: number of classes in the dataset
    :param size: size of the data set
    :param scatter: scatter parameter
    :return:
    '''
    x_data = []
    y_data = []
    centers = [[] for _ in range(classes)]
    for i in range(classes):
        for j in range(features):
            centers[i].append(np.random.randint(0,1000)/1000)
    for i in range(size):
        x_data.append([])
        c = np.random.randint(0,classes)
        #y_data.append(c)
        y_data.append([int(c==j) for j in range(classes)])
        for j in range(features):
            x_data[-1].append(centers[c][j] + np.random.normal()*scatter)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

if __name__ == "__main__":

    from sklearn.datasets import fetch_mldata
    import matplotlib.pyplot as plt

    print("loading data")
    mnist = fetch_mldata('MNIST original')
    inputX,inputY =[], []
    for i in range(70000):
        r = np.random.randint(0,70000)
        inputX.append(mnist['data'][r])
        inputY.append(mnist['target'][r])
    inputX = np.array(inputX)
    inputY = np.array(inputY)
    inputX=inputX.reshape(-1 , 784)
    inputY=inputY.astype(int)

    totalClasses = max(inputY)-min(inputY)+1
    temp = []
    for i in range(len(inputY)):
        temp.append([int(inputY[i] == j) for j in range(totalClasses)])
    temp = np.array(temp)
    temp = temp.reshape(len(inputY),-1)
    inputY = temp
    #inputX, inputY = createData('x*y',['x','y'],1000,10)
    #inputX, inputY = createDataClassification(10,10,1000,0.1)

    print("creating nn")
    nn = Neural([784, 30, 10], 0.1, 0.0001)
    nn.feed(inputX, inputY)
    for i in range(100):
        print("iteration", i)
        nn.learn()


    nn.forward(nn.datasets-1)
    print(nn.Y[nn.datasets-1], (nn.activation[nn.layers-1]))
    nn.forward(0)
    print(nn.Y[0], (nn.activation[nn.layers-1]))


    plt.plot(nn.J)
    plt.ion()
    plt.plot(nn.JTest)
    plt.pause(0.00001)
    plt.ioff()
    plt.figure("theta flow")
    plt.plot(nn.thetaFlow)
    plt.figure("activation flow")
    plt.plot(nn.activationFlow)
    plt.show()