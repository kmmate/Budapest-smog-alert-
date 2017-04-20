# (I): Print statistics on missing number of data on a day
# (II): Plot distance and correlation matrices
# (III): Impute the PM10 missing values based on t, in all Training, Testing, Evaluation periods


savefigure_mode=False  #if True, script saves figures but not display them, if False no saving but showing


#Importing dependencies
import pandas as pd
import numpy as np
import matplotlib as mpl
if savefigure_mode==True:
    mpl.use('pgf')
    #Credits to Bennett Kanukat http://bkanuka.com/articles/native-latex-plots/
    def figsize(scale):
        fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*golden_mean             # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size
    pgf_with_rc_fonts = {'font.family': 'serif', 'figure.figsize': [2, 3], 'pgf.texsystem': 'pdflatex'}
    mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from  geodistance import InvDistWeightsP
import seaborn as sns
import time
import csv

# To make LaTeX font work: 1) add the folder of MikTeX to the path of the project interpreter
# 2) as administrator, go to folder MikTeX/Maintainance (admin), open MikTeX Package manager, install doublestroke
#package

# Import the periods from excel
y_train=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_train.xlsx" , na_values='NaN')
y_train.name='y_train'
y_test=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_test.xlsx" , na_values='NaN')
y_test.name='y_test'
y_eval15=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_eval15.xlsx" , na_values='NaN')
y_eval15.name='y_eval15'
y_eval15_window=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_eval15_window.xlsx" ,
                              na_values='NaN')
y_eval15_window.name='y_eval15_window'
y_eval17=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_eval17.xlsx" , na_values='NaN')
y_eval17.name='y_eval17'
y_eval17_window=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_eval17_window.xlsx" ,
                              na_values='NaN')
y_eval17_window.name='y_eval17_window'
 #setting dates, index, numeric values
periodslist=[y_train, y_test, y_eval15, y_eval15_window, y_eval17, y_eval17_window]
for x in [y_train, y_test, y_eval15_window, y_eval17, y_eval17_window]:
    x['dates']=pd.to_datetime(x['dates'])
    x.index=x['dates']
    del x['dates']
    x.apply(func=pd.to_numeric)
y_eval15.apply(func=pd.to_numeric)

#Import the coordinates
coordinates=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Coordinates.xlsx", index=0)
coordinates=coordinates.apply(func=pd.to_numeric)
#Import the distancematrix
distancematrix=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Distancematrix.xlsx", index=0)
distancematrix=distancematrix.apply(func=pd.to_numeric)


# (I): Print statistics on missing number of data on a day

# Compute in each period for each date, how many stations have missing data,
# append to the period's DataFrame as a column, print it, then delete the column
for x in periodslist:
    x['NofNaNs']=x.isnull().sum(axis=1)
    print('In period', x.name, 'the maximum number of missing values at a date is', np.max(x['NofNaNs']), '\n',
          'In period', x.name, 'the mean number of missing values at a date is', np.mean(x['NofNaNs']), '\n',
          'In period', x.name, 'the std of the number of missing values at a date is', np.std(x['NofNaNs']), '\n')
    del x['NofNaNs']


# (II):
#  (II.1) Plot distance matrix
#  (II.2) Compute and plot correlation matrix


# (II.1): Plot distance matrix

# Rename stations for plotting
y_train_accentname=y_train.rename(columns={'Erzsebet': 'Erzsébet',
                                'Korakas': 'Kőrakás',
                                'Kosztolanyi': 'Kosztolányi',
                                'Pesthidegkut': 'Pesthidegkút',
                                'Szena': 'Széna'})
# Distancematrix, remove dropped stations
# list of kept stations
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Kept_Stations.csv", 'r') as myfile:
    rd = csv.reader(myfile, delimiter=',')
    # list the headers
    keptstations = list(rd)
    # correct for the returned list of list file only to have a simple a list
    keptstations = keptstations[0]
# define distancematrix with accents in names
distancematrix_accentname=distancematrix.copy()
# remove dropped stations
for key in distancematrix_accentname:
    if key not in keptstations:
        distancematrix_accentname.drop(key, inplace=True, axis=1)
        distancematrix_accentname.drop(key, inplace=True)
# rename the columns to names with accents
distancematrix_accentname=distancematrix_accentname.rename(columns={'Erzsebet': 'Erzsébet',
                                'Korakas': 'Kőrakás',
                                'Kosztolanyi': 'Kosztolányi',
                                'Pesthidegkut': 'Pesthidegkút',
                                'Szena': 'Széna'})
# set the indices to accented names
distancematrix_accentname.index=distancematrix_accentname.columns
# Plot the distancematrix
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax=sns.heatmap(distancematrix_accentname, cmap='Reds_r', square=True)
ax.collections[0].colorbar.set_label('Distance ($km$)', size=12)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45, ha='right')
locs, labels = plt.yticks()
plt.setp(labels, rotation=45)
plt.title('Distance matrix')
plt.tight_layout()
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(0.9)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\distmat.pgf', bbox_inches='tight')#, dip=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\distmat.png', bbox_inches='tight')#, dip=500
else:
    # show
    plt.show(block=True)

# (II.2): Compute and plot correlation matrix

#  Compute correlation matrix
corrmat=y_train.corr(method='pearson', min_periods=1)

# Plot correlation matrix with accents
corrmat_accentname=y_train_accentname.corr(method='pearson', min_periods=1)
print('The correlation between the series in the training period is \n', corrmat_accentname,'\n')
# Plot correlation matrix as heatmap, if savefigure_mode==True then save it it, otherwise show it
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax=sns.heatmap(corrmat_accentname, cmap='Reds', square=True)
ax.collections[0].colorbar.set_label('Pearson-$r$', size=12)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45, ha='right')
locs, labels = plt.yticks()
plt.setp(labels, rotation=45)
plt.title('Between-station correlation of PM$_{10}$ level -- training period')
plt.tight_layout()
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(1.1)[0], figsize(1.1)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\corrmat.pgf', bbox_inches='tight')#, dip=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\corrmat.png', bbox_inches='tight')#, dip=500
else:
    # show
    plt.show(block=True)


# (III) Impute missing values

# Create copies before imputing to visually check the results of imputation
y_train_copy=y_train.copy()
y_test_copy=y_test.copy()

#Imputing the missing values: based on either distance or correlation matrix
impute_mode='correlation' #if 'distance': inverse distance weighting, if 'correlation': correlation based

# For 'distance', the weights are derived with the help of InvDistWeights from 'geodist.py'; for 'correlation', the
# weights are derived with CorrWeights function defined below.

# InvDistWeightsP is fed with the inputs:
#(1) 'df': the DataFrame containing the distances
#(2) 's': for every date, every station (s: station)  the weights have to be determined based on
# the distance from the stations with no NaNs
#(3) 'keptlist': for every date a list of stations with no NaNs

#CorrWeights is fed with the inputs:
#(1) 'df': the DataFrame containing correlations
#(2) 's': for every date, every station (s: station) the weights have to be determined based on
# the correlation of the given station,s, with other stations of no NaNs
#(3) 'keptlist': for every daty a list of statins with no NaNs

#Define CorrWeights
def CorrWeights(df_corr, s, keptlist):
    """
    Returns an array of correlation based weights ('weights') for station 's' calculated with stations in 'keptlist'.
    :param df_corr: DataFrame containing the correlation matrix
    :param s: station for which the weights are to be determined
    :param keptlist: list of stations which contributes to the weights of 's'
    :return:
    """
    # Get the array of correlations of 's' with stations in 'keptlist'
    corr=df_corr.ix[keptlist,s].values
    # Convert the correlations to weights with corr2weights
    def corr2weights(corr):
        return corr/sum(corr)
    return corr2weights(corr)

# Impute
start_time=time.time()
#Looping through periods
for x in periodslist:
    #through dates
    for row in range(0, len(x)):
        #(3) determine the list of stations with no NaNs
        nonanstations=list()
        for key in x:
            if pd.notnull(x.ix[row, key]):
                nonanstations.append(key)
        # get the PM10 values for the non NaN stations for the given date
        nonanvalues=list()
        for stations in nonanstations:
            nonanvalues.append(x.ix[row,stations])
        #for stations
        for key in x:
            #if NaN,
            if pd.isnull(x.ix[row, key]):
                if impute_mode=='distance':
                    #calling InvDistWeightsP
                    weights=InvDistWeightsP(distancematrix, key, nonanstations)
                    #print('WeightsP', weightsP, '\n', 'sum', sum(weights))
                elif impute_mode=='correlation':
                    weights=CorrWeights(corrmat, key, nonanstations)
                x.ix[row, key]=np.dot(nonanvalues,weights)
                #print(x.ix[row,key])
    #export the imputed values to excel
    x.to_excel('C:\\Users\\Máté\\Dropbox\\CEU\\2017 Winter\\Econ\\Smog\\Data\\' + x.name + '_imputed.xlsx')
print('The time of looping is', time.time()-start_time)


# Visually check imputation results
plt.rc('text', usetex=True)
plt.rc('font',family='serif')
plt.plot(y_train['Csepel'],label='PM$_{10}$ level at Csepel imputed')
plt.title('PM$_{10}$ concentration in the training period')
#plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('PM$_{10}$ concentration')
plt.plot(y_train_copy['Csepel'],label='PM$_{10}$ level at Csepel')
#plt.title('PM$_{10}$ concentration in the eval17 period')
plt.legend(loc='upper right')
#plt.xlabel('Time')
#plt.ylabel('PM$_{10}$ concentration')
plt.show(block=True)