#For PM10:
# (I) Dissectioning into Training, Test and Evaluation (+Evaluation with window) periods
# (II): Providing descriptive statistics for each period
# (III): Correcting anomalies (detected with descriptives),
# (IV): Print the corrected, not imputed,  data to latex table and plot distributions, saving to pdf
# (V): Export the different periods (of corrected data) into excel files

savefigure_mode=True  #if True, script saves figures but not display them, if False no saving but showing

# Importing dependencies
import numpy as np
from scipy import stats
import pandas as pd
import datetime as dt
# Setting plotting formats
import matplotlib as mpl
if savefigure_mode==True:
    mpl.use('pgf')
    def figsize(scale):
        fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*golden_mean              # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size

    pgf_with_rc_fonts = {'font.family': 'serif', 'figure.figsize': figsize(0.9), 'pgf.texsystem': 'pdflatex'}
    mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
sns.set_style('whitegrid')


#Importing the data produced by dataimport.py
#Importing the data from excel
#importing
y=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Smog_PM10_stacked.xlsx", na_values='NaN')
#setting the date
y['dates']=pd.to_datetime(y['dates'])
#setting as index and delete column
y.index=y['dates']
del y['dates']
#convert all values to numbers
y=y.apply(func=pd.to_numeric)


#(I): Dissection into Training, Test and Evaluation periods

#Dissection into periods
#training
y_train=y.ix['2013-01-01':'2014-12-31']
y_train.name='y_train'
#testing
y_test=y.ix['2015-12-01':'2016-07-22']
y_test.name='y_test'
#evaluation 2015
y_eval15=y.ix['2015-11-08']
#as it is only one day it turns into series object, so correct for it:
y_eval15=pd.DataFrame(y_eval15)
#transpose the index and columns to get back the original structure
y_eval15=y_eval15.T
y_eval15.name='y_eval15'
#evaluation 2015 with window
y_eval15_window=y.ix['2015-11-03':'2015-11-13']
y_eval15_window.name='y_eval15_window'
#evaluation 2017
y_eval17=y.ix['2017-01-23':'2017-01-26']
y_eval17.name='y_eval17'
#evaluation 2017 window
y_eval17_window=y.ix['2017-01-18':'2017-01-31']
y_eval17_window.name='y_eval17_window'
#dataFrames to a list
periodslist=[y_train, y_test, y_eval15, y_eval15_window, y_eval17, y_eval17_window]


#(II): Providing descriptive statistics for each period

#Dictionary of DataFrames of descriptives
#create a dictionary (desc) which contains the DataFrames of descriptives of each period
desc=dict()
for x in periodslist:
    #for each stations
    desc[x.name+'_desc']=pd.DataFrame({key: [np.mean(x[key]),
                                             x.median()[key],
                                             stats.mode(x[key])[0][0],
                                             np.max(x[key]),
                                             np.min(x[key]),
                                             np.std(x[key]),
                                             x.count()[key],
                                             len(x[key])] for key in x},
                                      index=['Mean', 'Median' ,'Mode', 'Max', 'Min' , 'Std',
                                             'No. of observations', 'No. of days in the period'])
    #overall
    desc[x.name+'_desc']['Overall']=[np.mean(x.stack()),
                                     x.stack().median(),
                                     stats.mode(x.stack())[0][0],
                                     np.max(x.stack()),
                                     np.min(x.stack()),
                                     np.std(x.stack()),
                                     x.stack().count(),
                                     len(x.ix[:, 0])*len(x.columns)]
    #print
    print('For period',x.name, ' the descriptives are \n', desc[x.name+'_desc'], '\n')



#(III): correcting anomalies (detected with descriptives)

#Anomaly: at station 'Korakas' 0's are present instead of NaN; change it and count the instances
instances=0
pd.options.mode.chained_assignment = None #turning off warning
for x in periodslist:
    for row in x.index:
        if x.ix[row,'Korakas']==0:
            x.ix[row, 'Korakas']=np.NaN
            instances=instances+1
print('\n\n No. of days when 0''s are replaced by NaN''s at Korakas: ' ,instances, '\n\n')




#(IV): Print the corrected data to latex table and plot the distributions in Training and Test period

#(IV.1): Calculate the descriptives for the corrected data
#create a dictionary (desc) which contains the DataFrames of descriptives of each period
desc_corr=dict()
for x in periodslist:
    #for each stations
    desc_corr[x.name+'_desc']=pd.DataFrame({key: [np.mean(x[key]),
                                             x.median()[key],
                                             stats.mode(x[key])[0][0],
                                             np.max(x[key]),
                                             np.min(x[key]),
                                             np.std(x[key]),
                                             x.count()[key],
                                             len(x[key])] for key in x},
                                      index=['Mean', 'Median' ,'Mode', 'Max', 'Min' , 'Std',
                                             'No. of observations', 'No. of days in the period'])
    #overall
    desc_corr[x.name+'_desc']['Overall']=[np.mean(x.stack()),
                                     x.stack().median(),
                                     stats.mode(x.stack())[0][0],
                                     np.max(x.stack()),
                                     np.min(x.stack()),
                                     np.std(x.stack()),
                                     x.stack().count(),
                                     len(x.ix[:,0])*len(x.columns)]
    #print('For period', x.name, 'the corrected descriptives are \n', desc_corr[x.name+'_desc'], '\n')
pd.options.mode.chained_assignment ='warn' #turning back warning

#(IV.2): Printing into latex tables
#rounding function
def rounding(x):
    return x.round(decimals=4)
#latex table strings
latex_descr_train=desc_corr['y_train_desc'].apply(rounding).to_latex()
latex_descr_test=desc_corr['y_test_desc'].apply(rounding).to_latex()
latex_descr_eval15=desc_corr['y_eval15_desc'].apply(rounding).to_latex()
latex_descr_eval17=desc_corr['y_eval17_desc'].apply(rounding).to_latex()
#printing the latex table strings
print('######## \n\n LaTeX table for the corrected (not imputed) descriptives of y_train \n',
      latex_descr_train, '\n ########\n')
print('######## \n\n LaTeX table for the corrected (not imputed) descriptives of y_test \n',
      latex_descr_test, '\n ########\n')
print('######## \n\n LaTeX table for the corrected (not imputed) of y_eval15 \n',
      latex_descr_eval15, '\n ########\n')
print('######## \n\n LaTeX table for the corrected (not imputed) descriptives of y_eval17 \n',
      latex_descr_eval17, '\n ########\n')

#(IV.3) Plot the distributions of the corrected data in the Training and Test period
# Rename the stations for plotting, with accents
y_train_accentname=y_train.rename(columns={'Erzsebet': 'Erzsébet',
                                'Korakas': 'Kőrakás',
                                'Kosztolanyi': 'Kosztolányi',
                                'Pesthidegkut': 'Pesthidegkút',
                                'Szena': 'Széna'})
y_test_accentname=y_test.rename(columns={'Erzsebet': 'Erzsébet',
                                'Korakas': 'Kőrakás',
                                'Kosztolanyi': 'Kosztolányi',
                                'Pesthidegkut': 'Pesthidegkút',
                                'Szena': 'Széna'})
# Plotting formats
mpl.rc('xtick', labelsize=12)
mpl.rc('font', family='serif')
# Boxplot coloring format
color=dict(boxes='Blue', whiskers='Blue', medians='Red')
# Boxplot for Training
y_train_accentname.plot.box(color=color)
plt.title('Distributions of PM$_{10}$ level in the training period')
plt.ylabel('PM$_{10}$ level', fontsize=12)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.tight_layout()
# If savefigure_mode==True then save figure, otherwise plot it
if savefigure_mode==True:
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\boxplot_train.pgf', dpi=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\boxplot_train.png', dpi=500)
else:
    # show
    plt.show(block=True)

# Violinplot for Training
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax=sns.violinplot(data=y_train_accentname, inner='box', color='Red', saturation=0.8)
plt.title('Violin plot of PM$_{10}$ distributions -- training period')
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
#plt.tight_layout()
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(0.9)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\violin_train.pgf', bbox_inches='tight')#, dip=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\violin_train.png', bbox_inches='tight')#, dip=500
else:
    # show
    plt.show(block=True)

# Boxplot for Test
plt.clf()
y_test_accentname.plot.box(color=color)
plt.title('Distributions of PM$_{10}$ level in the test period')
plt.ylabel('PM$_{10}$ level', fontsize=12)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.tight_layout()
# If savefigure_mode==True then save, otherwise plot it
if savefigure_mode==True:
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\boxplot_test.pgf', dpi=300)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\boxplot_test.png', dpi=500)
else:
    # show
    plt.show(block=True)

# Violinplot for Test
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax=sns.violinplot(data=y_test_accentname, inner='box', color='Red', saturation=0.8)
plt.title('Violin plot of PM$_{10}$ distributions -- test period')
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
#plt.tight_layout()
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(0.9)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\violin_test.pgf', bbox_inches='tight')#, dip=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\violin_test.png', bbox_inches='tight')#, dip=500
else:
    # show
    plt.show(block=True)

#(V): export the different periods into excel files
for x in periodslist:
    x.to_excel("C:\\Users\\Máté\\Dropbox\\CEU\\2017 Winter\\Econ\\Smog\\Data\\"+x.name+'.xlsx', na_rep='NaN')