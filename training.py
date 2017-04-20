# (I): Compiles design matrix
# (II): Dissect the desing matrix into Training, Test and Eval periods
# (III): Train NeuralNet for each PM10 station

training_mode=False #if 'True' the training is done, if False, only the design matrix compiles

#Importing dependencies
import numpy as np
import  pandas as pd
import cloudpickle as cpickle
import matplotlib.pyplot as plt

if __name__=='__main__':
    # Importing
    # PM10 values
    y_train_pre=pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_train_imputed.xlsx')
    y_train_pre.index=y_train_pre['dates']
    del y_train_pre['dates']
    # Meteorological variables
    temp_avg=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_avg_imputed.xlsx", index='dates')
    temp_avg.name='temp_avg'
    temp_min=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_min_imputed.xlsx", index='dates')
    temp_min.name='temp_min'
    temp_max=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_max_imputed.xlsx", index='dates')
    temp_max.name='temp_max'
    wind_power=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\wind_power_imputed.xlsx", index='dates')
    wind_power.name='wind_power'
    wind_blow=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\wind_blow_imputed.xlsx", index='dates')
    wind_blow.name='wind_blow'

    #(I): Design matrix
    # Merge the meteo data to one DataFrame
    X_pre=pd.concat([temp_avg, temp_min, temp_max, wind_power, wind_blow],axis=1)
    #drop wind_power as it barely has any variation
    del X_pre['wind_power']


    # Work with two matrices:
    # (I.1) define X_l, which contains the lagged variables originally in X_pre
    # (I.2) define X_oth which contains other variables.
    # (I.3): Concatinate them to have the design matrix

    #(I.1)
    # Create a function which retuns the DataFrame of lagged values to be concatinated with X
    def laggedDF(df,lags):
        """
        Takes DataFrame of sorted time series data, and returns a DataFrame containing the lagged time series.
            Columns are named after X.columns+'_lag%lag'.

        #Dependencies: pandas
        :param df: DataFrame of sorted time series
        :param lags: number of lags to produce (scalar, range)
        :return:
        """
        #empty DataFrame
        out=pd.DataFrame()
        for lag in lags:
            #actual lagged DF
            actuallag=df.shift(periods=lag)
            #rename columns
            for key in df:
                actuallag=actuallag.rename(columns={key: key+'_lag'+str(lag)})
            #add to overall lagged
            out=pd.concat([out, actuallag],axis=1)
        return out

    # Call it for X to create X_l
    X_l=laggedDF(X_pre, range(1,5))

    #(I.2) Other variables
    #Create a function for dynamic variance
    def dynVar(df,lag):
        """
        Takes a DataFrame and returns a DataFrame with the dynamic var (deviation from the dynamic mean)

        Dependencies: pandas, numpy
        :param df: a DataFrame
        :param lag: the lagging of the dynamic sum of squared deviation, inclusive interval: [t-lag,t]
        :return: a DataFrame with renamed columns
        """
        #empty dict
        out=dict()
        for key in df:
            sd=[np.nan]*lag
            for row in range(lag,len(df)):
                sd.append((np.std(df.ix[(row-lag):row, key]))**2)
            out[key + '_dynVar']=sd
        return pd.DataFrame(out, index=df.index)

    # Call dynSD function to create X_oth
    X_oth=dynVar(X_pre,7)

    #(I.3) Concatinate
    X=pd.concat([X_pre, X_l,X_oth], axis=1)



    # (II): Dissect into Training, Test and Eval periods, correct y_train_pre for the lags, write to excel
    #train
    X_train=X['2013-01-08':'2014-12-31']
    y_train=y_train_pre['2013-01-08':]
    # test
    X_test = X.ix['2015-12-01':'2016-07-22']
    # evaluation 2015
    X_eval15_window = X.ix['2015-11-03':'2015-11-13']
    # evaluation 2017
    X_eval17_window = X.ix['2017-01-18':'2017-01-31']
    # Normalisation
    #train
    y_train=(y_train-y_train.mean())/y_train.std()
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_train.name = 'X_train'
    # eval15
    X_eval15_window=(X_eval15_window-X_test.mean())/X_test.std()
    X_eval15_window.name = 'X_eval15_window'
    # eval17
    X_eval17_window = (X_eval17_window - X_test.mean()) / X_test.std()
    X_eval17_window.name = 'X_eval17_window'
    # test
    X_test = (X_test - X_test.mean()) / X_test.std()
    X_test.name = 'X_test'
    # dataFrames to a list
    periodslist_X = [X_train, X_test, X_eval15_window, X_eval17_window]
    # Writing to excel
    for x in periodslist_X:
        x.to_excel("C:\\Users\\Máté\\Dropbox\\CEU\\2017 Winter\\Econ\\Smog\\Data\\" + x.name + '.xlsx', na_rep='NaN')

if training_mode==True:
    #(II): Training the NeuralNet:
    # (II.1): import NeuralNet, write a function which accepts NeuralNet function, the data, and trains the Net
    # (II.2): call the function with the data to train NeuralNet


    #(II.1): Training function
    #Import
    from neuralnet import NeuralNet

    # Some test of NeuralNet
    dw={'W_hid': np.array([1,2,3,0.1,1.01,2.0]), 'W_out': np.array([9,8,10])}
    x_vmi=np.array([9,8,1])
    #deriv=NeuralNet(x_vmi,dw,True, 26)
    #print(deriv)

    # Test of training for one station
    #X_fasz=X_train.as_matrix()
    #print(X_fasz[0,:], X_train.ix['2013-01-08'])

    #Training function
    def NeuralNet_train(yvec, Xmatrix, M=2, batchsize=10, learnrate=0.02, penalty=1, iternumber=10):
        """
        Trains the NeuralNet (in neuralnet.py) parameters with regularirised stochastic mini-batch gradient descent
        :param yvec: vector of actual y as numpy array
        :param Xmatrix: design matrix without constant as numpy array
        :param M: number of neurons in the hidden layer, without constant
        :param learnrate:
        :param iternumber: number of iterations
        :return:
        """
        # Import dependencies
        import numpy as np
        from neuralnet import NeuralNet

        #Append the constant/"bias" to X
        Xb=np.insert(Xmatrix, Xmatrix.shape[1], 1, axis=1)

        # Training
        # default random generator
        np.random.seed(1)
        # create copies of y and X which will be reshuffled
        yp=yvec.copy()
        Xp=Xb.copy()
        # Initialise weights randomly from N(0,1)
        W_hid=np.random.randn(M*Xp.shape[1])
        W_out=np.random.randn((M+1))
        print('Initial W_hid=', W_hid, '\n')
        print('Initial W_out=', W_out, '\n')
        # put into dictionary for NeuralNet
        w={'W_hid': W_hid, 'W_out': W_out}
        # Iteration
        for iter in range(iternumber):
            print('No. of iteration:', iter + 1, '\n\n')
            # (1) Reshuffle (permute) observations
            p=np.random.permutation(len(yp))
            # permute y
            yp=yp[p]
            # permute X
            Xp=Xp[p]

            # (2) Iteration through time/mini batches
            for tau in range(0, len(yp)-batchsize+1, batchsize):
                # (3) Call NeuralNet in 'gradientmode' for each observation in the given mini-batch, compute the sum of gradients
                # in the given mini-batch
                # numpy array in which summed up are the gradients
                sumgradW_hid = np.zeros(M * Xp.shape[1])
                sumgradW_out = np.zeros(M + 1)
                # summation
                for t in range(tau, tau+batchsize-1):
                    # dictionary of the current gradient
                    current_gradient=NeuralNet(Xp[t,:], w, True, yp[t])
                    # adding up the gradients
                    sumgradW_hid=sumgradW_hid+current_gradient['dW_hid']
                    sumgradW_out=sumgradW_out+current_gradient['dW_out']
                    #print(sumgradW_out)
                # update W_hid
                W_hid=W_hid-learnrate*((1/batchsize)*sumgradW_hid+penalty*2*W_hid)
                # update W_out
                W_out=W_out-learnrate*((1/batchsize)*sumgradW_out+penalty*2*W_out)
                # update the dictionary
                w = {'W_hid': W_hid, 'W_out': W_out}
                #print the Euclidean norm of the gradients
                #print('The L2 norm of sumgradW_hid:\n', np.dot(sumgradW_hid,sumgradW_hid),'\n L2 norm of sumgradW_out:',
                    #  np.dot(sumgradW_out, sumgradW_out), '\nThe sumgradW_out:\n', sumgradW_out,'\n')
                #print the updated W_out
                #print('W_out=', w['W_out'], '\n')
        if iter==(iternumber-1):
            w_out={'W_hid': np.array([np.float(row) for row in W_hid]),
                   'W_out': np.array([np.float(row) for row in W_out])}
        return w_out

    #(II.2): Call the training function for stations
    # Design matrix to numpy array, augment with constant
    X_np=X_train.values
    #add constant
    X_npb=np.insert(X_np, X_np.shape[1], 1, axis=1)

    # Train: calling NeuralNet_train for every station, save the weights into a dictionary, then save dictionary
    W_hat=dict()
    y_train_hat=dict()
    for key in y_train:
        print('\n\n Station:', key, ' the optimisation started\n')
        # Estimated weights
        W_hat[key]=NeuralNet_train(y_train[key].values,
                            X_train.values,
                            M=4, iternumber=30, penalty=0, learnrate=0.5, batchsize=241)
        # Saving the estimated weights
        with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\W_hat.p", 'wb') as myfile:
            cpickle.dump(W_hat, myfile)
        #Predict
        yhat=np.array([])
        for t in range(0,len(y_train)):
            yhat=np.append(yhat, np.float(NeuralNet(X_npb[t,:], W_hat[key], False)))
        y_train_hat[key]=yhat
        y_train_hat=pd.DataFrame(y_train_hat, index=y_train.index)

        # Plot
        f=plt.figure()
        plt.plot(y_train[key], label='original')
        plt.plot(y_train_hat[key], label='fitted')
        plt.title(key)
        plt.savefig('C:\\Users\\Máté\\Dropbox\\CEU\\2017 Winter\\Econ\\Smog\\latex\\'+key+'_yhat.png', bbox_inches='tight')

