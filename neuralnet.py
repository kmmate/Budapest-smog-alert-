# Neural Network function with 1 hidden layer
def NeuralNet(dvector, weights, gradientmode, y=None):
    """
    Neural Network function, with 1 hidden layer of M neurons. Can be called to either retrive the predicted value
    or the gradients. Note: in the 'dvector' argument, the constant 1 has to be at the last element

    Dependencies: numpy, bigfloat
    :param dvector: design vector without constant as numpy.array
    :param weights: dictionary with entries:
                     1. 'W_hid': np array of size (1)x(K*M), where K=(#of variables with constant), 'm'th 1*K elements
                      belong to the weight vector going to the 'm'th neuron in the hidden layer
                     2. 'W_out': np array of size (1)x(M+1), where each row corresponds to the weight from the 'm'th
                      neuron and a "bias" term
    :param gradientmode: if set True, the function returns gradients: 'y' has to be the given for this functionality
    :param y: actual y, need for gradientmode, not for prediction
    :return d or yhat: predicted scalar value or list of gradients with ordered elements:
                     1. ('dW_hid'): (1)x(K*M) stacked gradients of weight vectors going to W_hid: 'm'th 1*K elements
                      belong to the weight vector going to the 'm'th neuron in the hidden layer
                     2. ('dW_out'): (1)x(M+1) gradient vector of weights in 'W_out'

    """
    # import dependencies
    import numpy as np
    import bigfloat as bfloat

    # Define the type of function in hidden layer
    def sigmoidx(x, derivative):
        # the function
        s=1/(1+bfloat.exp(-x/20, bfloat.precision(30)))
        # if derivative is true, the output is the derivative
        if derivative==True:
            return (s+s*(1-s))
        else:
            return s

    #(I): Compute yhat
    # evaluate the neurons in the hidden layer, as vector-valued functions
    f=np.array([sigmoidx(np.dot(dvector, weights['W_hid'][k:k+len(dvector)]), False)
                for k in range(0,len(weights['W_hid']), len(dvector))])
    # augment it with "bias" and compute yhat
    fb=np.append(f, 1)
    #print('Hidden layer=', fb, '\n')
    #calculate yhat
    yhat=np.dot(weights['W_out'], fb)
    #print('yhat=', yhat, '\n')

    #(II): When gradientmode is activated, return the two gradientvectors, assuming loss function L=0.5*(yhat-y)^2
    if gradientmode==True:
        #if actual y is not given, print error
        if y is None:
            print('Actual y has to be given for ''gradientmode''.\n')
        #vector of derivatives of the elements of f except the "bias"
        df=np.array([sigmoidx(np.dot(dvector, weights['W_hid'][k:k+len(dvector)]), True)
                    for k in range(0, len(weights['W_hid']), len(dvector))])
        # add constant:
        df=np.append(df, 1)
        #print('\n W_dis', weights['W_hid'], '\n')
        #print('Input to sigmoidx:\n', np.array([np.dot(dvector, weights['W_hid'][k:k+len(dvector)]) for k in range(0, len(weights['W_hid']), len(dvector))]), '\n')
        # gradient for W_hid:
        dW_hid=(yhat-y)*np.array([(weights['W_out'][m]*df[m]*dvector) for m in range(0, len(weights['W_out'])-1)])

        # flatten the array
        dW_hid=dW_hid.flatten()
        # gradient for W_out:
        dW_out=(yhat-y)*df
        # return results as dictionary
        gradient={'dW_hid': dW_hid, 'dW_out': dW_out}
        return gradient

    else:
        return yhat