'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#activation functions
def reLu(x):
    return np.maximum(x, 0)

def reLu_deriv(x):
    return np.where(x <= 0, 0, 1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

#loss functions
def mse(pred, real): #mean squared error
    n_samples = real.shape[0]
    loss = (1 / (2 * n_samples)) * np.sum(np.square(real - pred))
    return loss

def mse_deriv(pred, real):
    n_samples = real.shape[0]
    res = -1 / n_samples * (real - pred)
    return res

#weights initialization
def weight_param_init(in_layer, out_layer, ini_type='plain'):
    if ini_type == 'plain':
        return np.random.randn(in_layer, out_layer) *0.01
    elif ini_type == 'he':
        return np.random.randn(in_layer, out_layer)*np.sqrt(2/(in_layer + out_layer))
    elif ini_type == 'xavier':
        return np.random.randn(in_layer, out_layer) / (np.sqrt(in_layer + out_layer))

np.random.seed(0)

class NN:

    ''' X and Y are dataframes '''

    def __init__(self, neurons_per_layer, lr, epochs=500): #initialize number of neurons per layer, epochs and learining rate(lr)
        self.neurons_layer1 = neurons_per_layer[0]
        self.neurons_layer2 = neurons_per_layer[1]
        self.neurons_output_layer = neurons_per_layer[2]
        self.epochs = epochs
        self.lr = lr
        self.m = {}  # init 1st moment (Adam optimizer)
        self.v = {}  # init 2nd moment (Adam optimizer)

    def init_layers(self, X, y):
        neurons_layer1 = self.neurons_layer1
        neurons_layer2 = self.neurons_layer2
        input_dim = X.shape[1]
        output_dim = self.neurons_output_layer

        #initializing neurons with random values using xavier initialization
        self.w1 = weight_param_init(input_dim, neurons_layer1, ini_type='xavier') # weights
        self.b1 = np.zeros((1, neurons_layer1)) # biases
        self.w2 = weight_param_init(neurons_layer1, neurons_layer2, ini_type='xavier')
        self.b2 = np.zeros((1, neurons_layer2))
        self.w3 = weight_param_init(neurons_layer2, output_dim, ini_type='xavier')
        self.b3 = np.zeros((1, output_dim))

        self.init_adam_params()

        self.x = X
        self.y = y

    def feed_forward(self):
        #z = [(output from previous layer) * w] + b
        #a = reLu(z) or sig(z)
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = reLu(z1) #hidden layer 1
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = reLu(z2) #hidden layer 2
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(z3) #output layer

    def back_propagation(self):
        loss = mse(self.a3, self.y)
        print('Error :', loss)
        #chain rule
        a3_delta = mse_deriv(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * reLu_deriv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * reLu_deriv(self.a1) # w1
        self.a = [a1_delta, a2_delta, a3_delta]
        self.get_w_grad()
        self.get_b_grad()

    def get_w_grad(self): #gradient of weights
        a1_delta, a2_delta, a3_delta = self.a
        dw3 = np.dot(self.a2.T, a3_delta)
        dw2 = np.dot(self.a1.T, a2_delta)
        dw1 = np.dot(self.x.T, a1_delta)
        self.grad_w = [dw1, dw2, dw3] #weights gradient list

    def get_b_grad(self): #gradient of biases
        a1_delta, a2_delta, a3_delta = self.a
        db3 = np.sum(a3_delta, axis=0, keepdims=True)
        db2 = np.sum(a2_delta, axis=0)
        db1 = np.sum(a1_delta, axis=0)
        self.grad_b = [db1, db2, db3] #biases gradient list

    def gd(self): #gradient descent
        dw1, dw2, dw3 = self.grad_w
        db1, db2, db3 = self.grad_b

        # weights and bias correction
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1


    def init_adam_params(self): #initialize adam parameters(moments m & v)
        self.m["dw1"] = np.zeros_like(self.w1)
        self.m["dw2"] = np.zeros_like(self.w2)
        self.m["dw3"] = np.zeros_like(self.w3)

        self.m["db1"] = np.zeros_like(self.b1)
        self.m["db2"] = np.zeros_like(self.b2)
        self.m["db3"] = np.zeros_like(self.b3)

        self.v["dw1"] = np.zeros_like(self.w1)
        self.v["dw2"] = np.zeros_like(self.w2)
        self.v["dw3"] = np.zeros_like(self.w3)

        self.v["db1"] = np.zeros_like(self.b1)
        self.v["db2"] = np.zeros_like(self.b2)
        self.v["db3"] = np.zeros_like(self.b3)

    def adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8, t=0): #adam optimizer
        dw1, dw2, dw3 = self.grad_w
        db1, db2, db3 = self.grad_b

        mhat = {}
        vhat = {}

        # moving averages of 1st moment
        self.m["dw1"] = beta1*self.m["dw1"] + (1-beta1)*dw1
        self.m["dw2"] = beta1*self.m["dw2"] + (1-beta1)*dw2
        self.m["dw3"] = beta1*self.m["dw3"] + (1-beta1)*dw3

        self.m["db1"] = beta1*self.m["db1"] + (1-beta1)*db1
        self.m["db2"] = beta1*self.m["db2"] + (1-beta1)*db2
        self.m["db3"] = beta1*self.m["db3"] + (1-beta1)*db3

        # bias corrected estimator for 1st moment
        mhat["dw1"] = self.m["dw1"]/(1-np.power(beta1, t))
        mhat["dw2"] = self.m["dw2"]/(1-np.power(beta1, t))
        mhat["dw3"] = self.m["dw3"]/(1-np.power(beta1, t))

        mhat["db1"] = self.m["db1"]/(1-np.power(beta1, t))
        mhat["db2"] = self.m["db2"]/(1-np.power(beta1, t))
        mhat["db3"] = self.m["db3"]/(1-np.power(beta1, t))

        # moving averages of 2nd moment
        self.v["dw1"] = beta2*self.v["dw1"] + (1-beta2)*np.square(dw1)
        self.v["dw2"] = beta2*self.v["dw2"] + (1-beta2)*np.square(dw2)
        self.v["dw3"] = beta2*self.v["dw3"] + (1-beta2)*np.square(dw3)

        self.v["db1"] = beta2*self.v["db1"] + (1-beta2)*np.square(db1)
        self.v["db2"] = beta2*self.v["db2"] + (1-beta2)*np.square(db2)
        self.v["db3"] = beta2*self.v["db3"] + (1-beta2)*np.square(db3)

        # bias corrected estimator for 2nd moment
        vhat["dw1"] = self.v["dw1"]/(1-np.power(beta2, t))
        vhat["dw2"] = self.v["dw2"]/(1-np.power(beta2, t))
        vhat["dw3"] = self.v["dw3"]/(1-np.power(beta2, t))

        vhat["db1"] = self.v["db1"]/(1-np.power(beta2, t))
        vhat["db2"] = self.v["db2"]/(1-np.power(beta2, t))
        vhat["db3"] = self.v["db3"]/(1-np.power(beta2, t))

        # weights and bias correction
        self.w1 -= self.lr*(mhat["dw1"]/(np.sqrt(vhat["dw1"])+epsilon))
        self.w2 -= self.lr*(mhat["dw2"]/(np.sqrt(vhat["dw2"])+epsilon))
        self.w3 -= self.lr*(mhat["dw3"]/(np.sqrt(vhat["dw3"])+epsilon))

        self.b1 -= self.lr*(mhat["db1"]/(np.sqrt(vhat["db1"])+epsilon))
        self.b2 -= self.lr*(mhat["db2"]/(np.sqrt(vhat["db2"])+epsilon))
        self.b3 -= self.lr*(mhat["db3"]/(np.sqrt(vhat["db3"])+epsilon))


    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        self.init_layers(X, Y)
        for i in range(self.epochs):
            self.feed_forward()
            self.back_propagation()
            self.adam(t=i+1)
            #uncomment the below line and comment above line to change optimzer
            #self.gd()

    def predict(self,X):

        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values

        yhat is a list of the predicted value for df X
        """
        self.x = X
        self.feed_forward()
        yhat = self.a3 # a3 = output of last layer i.e output layer

        return yhat

    def CM(self, y_test,y_test_obs):
        '''
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        accuracy = (tp + tn)/(tp + tn + fp + fn)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print("Accuracy:", accuracy)
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")



df = pd.read_csv("LBW_cleaned.csv")

X = ["Community","Age", "IFA", "Weight", "Delivery phase", "BP"] #independent variables
y = ["Result"] #target variable

x_train, x_test, y_train, y_test = train_test_split(df[X], df[y],  test_size=0.3,  random_state=1, stratify=None)

model = NN([128, 32, 1], epochs=500, lr=0.0005)

model.fit(np.array(x_train), np.array(y_train))

yhat = model.predict(x_test)
model.CM(np.array(y_test), yhat)
