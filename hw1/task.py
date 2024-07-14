import numpy as np

def random_normal_weight_init(indim, outdim):
    return np.random.normal(0,1,(indim, outdim))

def random_weight_init(indim, outdim):
    b = np.sqrt(6)/np.sqrt(indim+outdim)
    return np.random.uniform(-b,b,(indim, outdim))

def zeros_bias_init(outdim):
    return np.zeros((outdim,1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(10)]for lab in labels], dtype=np.float32)


class Transform:
    """
    This is the base class. You do not need to change anything.

    Read the comments in this class carefully. 
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass



class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)
        self.mask = None

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self, x, train=True):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, grad_wrt_out):
        return self._sigmoid(self.x)*(1-self._sigmoid(self.x)) * grad_wrt_out



class LinearMap(Transform):
    """
    Implement this class
    feel free to use random_xxx_init() functions given on top
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        Transform.__init__(self)
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        self.alpha = alpha
        self.lr = lr
        self.indim = indim
        self.outdim = outdim

        # initialization of W and b
        self.W = random_normal_weight_init(self.indim, self.outdim)
        self.b = zeros_bias_init(self.outdim)
        self.zerograd()

        # for momentum update 
        self.momemtumW, self.momemtumB = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        self.x = x #record the input.
        return self.x @ self.W + self.b.T


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, outdim)
        return shape (batch_size, indim)
        Your backward call should Accumulate gradients.
        """
        assert self.x is not None, 'forward() need to be executed before backward().'
        batch_size, _ = grad_wrt_out.shape

        # dJ/dW, divided by the batch size.
        self.grad_w += 1/batch_size*(self.x.T @ grad_wrt_out)

        # dJ/db, divided by the batch size.
        self.grad_b += 1/batch_size*np.sum(grad_wrt_out, axis = 0, keepdims = True).T

        # dJ/dx
        return grad_wrt_out @ self.W.T


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        # general logic for momentum
        # new_change = learning_rate * gradient + alpha * momentum
        # update params with new_change
        # momentum = new_change
        # W update
        deltaW, deltaB = self.grad_w*self.lr+ self.alpha*self.momemtumW, self.grad_b*self.lr+ self.alpha*self.momemtumB
        self.W -= deltaW

        # b update
        self.b -= deltaB
        self.momemtumW, self.momemtumB = deltaW, deltaB
        

    def zerograd(self):
    # reset parameters
        self.grad_w = np.zeros((self.indim, self.outdim))
        self.grad_b = np.zeros((self.outdim, 1))
        # self.x = None # Why not this feature?

    def getW(self):
    # return weights
        return self.W

    def getb(self):
    # return bias
        return self.b

    def loadparams(self, w, b):
    # Used for Autograder. Do not change.
        assert self.W.shape == w.shape and self.b.shape == b.shape, f'wrong loading! W: {self.W.shape} vs {w.shape} | b: {self.b.shape} vs {b.shape}'
        self.W, self.b = w, b



class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels, train = True):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        batch_size, _ = logits.shape
        self.labels = labels
        self.y_hat = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        # if train:
        #     print(self.y_hat)
        loss = -1/batch_size*np.sum(np.sum(labels*np.log(self.y_hat),axis=1))
        return loss
    
    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        grad_wrt_out = self.y_hat - self.labels
        return grad_wrt_out

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        return (np.argmax(self.y_hat, axis=1) == np.argmax(self.labels, axis=1)).astype(int).mean()



class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, inp, outp, hiddenlayer=100, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.linear1 = LinearMap(indim=inp, outdim=hiddenlayer, alpha=alpha, lr=lr)
        self.relu1 = ReLU()
        self.linear2 = LinearMap(indim=hiddenlayer, outdim=outp, alpha=alpha, lr=lr)

    def forward(self, x, train=True):
    # x shape (batch_size, indim)
        return self.linear2.forward(self.relu1.forward(self.linear1.forward(x)))
        

    def backward(self, grad_wrt_out):
        grad1 = self.linear2.backward(grad_wrt_out)
        grad2 = self.relu1.backward(grad1)
        grad3 = self.linear1.backward(grad2)
        return grad3

    def step(self):
        self.linear2.step()
        self.linear1.step()

    def zerograd(self):
        self.linear2.zerograd()
        self.linear1.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        self.linear1.loadparams(Ws[0], bs[0])
        self.linear2.loadparams(Ws[1], bs[1])

    def getWs(self):
        """
        Return the weights for each layer
        You need to implement this. 
        Return weights for first layer then second and so on...
        """
        return [self.linear1.getW(), self.linear2.getW()]

    def getbs(self):
        """
        Return the biases for each layer
        You need to implement this. 
        Return bias for first layer then second and so on...
        """
        return [self.linear1.getb(), self.linear2.getb()]



class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    pass



class Dropout(Transform):
    """
    Implement this class
    """
    def __init__(self, p=0.5):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p = p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial
        Scale your output accordingly
        During test time, you should not apply any mask or scaling.
        """
        pass


    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        assert self.mask is not None, 'forward() need to be executed before backward().'
        pass





if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    from tqdm import tqdm 
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    X, y = data['data'] / 16.0, data['target']
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)

    trainLabels = labels2onehot(trainy)
    testLabels = labels2onehot(testy)

    indim = 64
    outdim = 10

    # set the hyper-params for training
    epoch = 50
    batch_size = 32
    lr = 1
    hiddim = 300
    alpha = 0.2

    # model instantiation
    model = SingleLayerMLP(indim, outdim, hiddenlayer=hiddim, alpha=alpha, lr=lr)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    for ep in range(epoch):
        # Shuffle train data
        assert len(trainX) == len(trainLabels)
        p = np.random.permutation(len(trainX))
        trainX_shuffled, trainLabels_shuffled = trainX[p], trainLabels[p]

        acc_temp = 0
        loss_temp = 0
        steps = len(trainX) // batch_size + 1
        for i in tqdm(range(steps)):
            train_temp = trainX_shuffled[i*batch_size : (i+1)*batch_size]
            label_temp = trainLabels_shuffled[i*batch_size : (i+1)*batch_size]
            cross_ent = SoftmaxCrossEntropyLoss()

            # forward the model
            logits = model.forward(train_temp)
            # calculate the loss
            loss = cross_ent.forward(logits, label_temp)
            # fetch the gradient of the loss function
            loss_grad = cross_ent.backward()
            # backward the model
            model.backward(loss_grad)
            # update the model
            model.step()
            # re-init the gradients
            # log the accumulated loss and accuracy
            model.zerograd()
            loss_temp += loss
            acc_temp += cross_ent.getAccu()
        # append the accumulated loss and accuracy to train_acc and train_loss
        train_acc.append(acc_temp/steps)
        train_loss.append(loss_temp/steps)

        test_loss_temp, tess_acc_temp = 0, 0
        # calculate the acc and loss on the test data
        # and append the results to test_acc and test_loss
        test_steps = len(testX) // batch_size + 1
        for i in tqdm(range(test_steps)):
            test_temp = testX[i*batch_size : (i+1)*batch_size]
            label_temp = testLabels[i*batch_size : (i+1)*batch_size]
            cross_ent = SoftmaxCrossEntropyLoss()
            logits = model.forward(test_temp)
            loss = cross_ent.forward(logits, label_temp, train = False)
            # no updates at test time
            model.zerograd()
            test_loss_temp += loss
            tess_acc_temp += cross_ent.getAccu()
        test_acc.append(tess_acc_temp/test_steps)
        test_loss.append(test_loss_temp/test_steps)

        print(f'epoch[{ep}]')
        print(f'losses | train=[{train_loss[-1]:.4f}] | test=[{test_loss[-1]:.4f}]')
        print(f'accs | train=[{train_acc[-1]:.2%}] | test=[{test_acc[-1]:.2%}]')

    import matplotlib.pyplot as plt
    epochs = range(epoch)
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, test_acc, 'b', label='Test accuracy')
    plt.title('Training and Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()