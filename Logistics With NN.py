# =============================================================================
# =============================================================================
# # # Logistics Regression with a Neural Network Mindset - By Michael Olugbenle 
# ***Project from Deep learning AI, by Andrew Ng
# =============================================================================
# =============================================================================
# The general Architecture of the learning algorithm
#   1.  Model Structure
#   2. Initialize the model's Parameters
#   3. Loop: Loss, current gradient(back prop), update Parameters

# =============================================================================
# # import all neccesary packages
# =============================================================================
#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.misc 
from PIL import Image
from scipy import ndimage
from matplotlib.pyplot import imread

train_dataset = h5py.File('train_catvnoncat.h5', "r")
#%%

# loading the dataset 
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
#%%

# Loading the dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


#%%

# =============================================================================
# #       Exploratory Data Analysis
# =============================================================================
# Example of a picture
index = 25
a = plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a ' " + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")
plt.show()


# number of training and test examples  and pixels height, weight, and channel 
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

print (f"# of training examples: m_train = {m_train}")
print (f"# of test examples: m_test = {m_test}")
print (f"weight and height and # of channels respectively = {num_px}")
print (f"Shape of train_set_x = {train_set_x_orig.shape}")
print (f"Shape of train_set_y = {train_set_y.shape}")
print (f"Shape of test_set_x = {test_set_x_orig.shape}")
print (f"Shape of test_set_y = {test_set_y.shape}")

#%%
# Reshaping and Flattening the image
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten =  test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
#%%
# Standardize the dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#%%
# Define Activation(sigmiod)

# sigmoid(z) = 1/1+exp(-z) - Th sigmoid receives a numerical input and output a number between 0 and 1.

def sigmoid(z):
    # calculate the sigmoid of Z(a scalar or numpy array)
    s = 1/(1 + np.exp(-z))

    return s
#%%
# Next step is to initialize the model parameters

def init_params(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b
#%%
    # Forward and Backward Propagation
def propagate(w, b, X, Y):
    # num of training samples
    m = X.shape[1]

    # forward pass
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -1./m* np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    # back propagation 
    dw = (1/m)*(np.dot(X, (A-Y).T))
    db = (1/m)*(np.sum(A-Y))
    
    

    #cost = np.squeeze(cost)

    # gradient dictionary
    grads = {"dw": dw, "db": db}

    return grads, cost 

#%%
# Optimization 
    
def optimize(w, b, X, Y, epochs, lr, print_cost = False):
    costs = []
    for i in range(epochs):
        # calculate gradients
        grads, cost = propagate(w, b, X, Y)

        # get gradients
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - (lr*dw)
        b = b - (lr*db)
        
        if i % 100 == 0:
            costs.append(cost)
            print(f"cost after {i} epochs: {cost}")
        
        
            

    # param dict
    params = {"w": w, "b": b}

    # gradient dict
    grads  = {"dw": dw, "db": db}

    return params, grads, costs
#%%

# Predict
    
def predict(w, b, X):
    m = X.shape[1]
    Y_predict = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_predict[0, i] = 0
        else:
            Y_predict[0,i]  = 1

    return Y_predict 

#%%
    # Model

def model(X_train, Y_train, X_test, Y_test, epochs, lr, print_cost = False):
    w, b = init_params(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr, print_cost = print_cost)

    w = params["w"]
    b = params["b"]

    Y_predict_train = predict(w, b, X_train)
    Y_predict_test  = predict(w, b, X_test)

    print("train_accuracy: {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test_accuracy : {} %".format(100-np.mean(np.abs(Y_predict_test  - Y_test)) * 100))

    log_reg_model = {"costs": costs,
                     "Y_predict_test": Y_predict_test, 
                     "Y_predict_train" : Y_predict_train, 
                     "w" : w, 
                     "b" : b,
                     "learning_rate" : lr,
                     "epochs": epochs}

    return log_reg_model

#%%

# activate the logistic regression model
myModel = model(train_set_x, train_set_y, test_set_x, test_set_y, epochs = 2000, lr = 0.005, print_cost = True)
#%%
img = Image.open("download.jpg")
img.load()
data = np.asarray(img, dtype = 'int32')
print(data)
#%%
  # change this to the name of your image file 
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
data.flatten()
data1 = np.resize(data, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T


my_predicted_image = predict(d["w"], d["b"], data1)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    







