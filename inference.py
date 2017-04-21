# (I): Predict the PM10 level and the error term for each period, make violinplots for residuals
#   in train and test periods
# (II): Compute confidence intervals

savefigure_mode=True #if True, script saves figures but not display them, if False no saving but showing

# Importing dependencies
import numpy as np
import  pandas as pd
import cloudpickle as cpickle
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
import seaborn as sns
sns.set_style('whitegrid')
from neuralnet import NeuralNet


# Import PM10 data and design matrices
# PM10 data
# train - import
y_train_pre = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_train_imputed.xlsx')
y_train_pre.index = y_train_pre['dates']
del y_train_pre['dates']
# train - correct for lags
y_train = y_train_pre['2013-01-08':]
# train - normalise
y_train = (y_train - y_train.mean()) / y_train.std()
# test - import
y_test = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_test_imputed.xlsx')
y_test.index = y_test['dates']
del y_test['dates']
# eval15 window
y_eval15_window = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_eval15_window_imputed.xlsx')
y_eval15_window.index = y_eval15_window['dates']
del y_eval15_window['dates']
# eval17 window
y_eval17_window = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\y_eval17_window_imputed.xlsx')
y_eval17_window.index = y_eval17_window['dates']
del y_eval17_window['dates']
# Standardisation
# eval15 window - normalise with test data
y_eval17_window = (y_eval17_window - y_eval17_window.mean()) / y_eval17_window.std()
# eval15 window - normalise with test data
y_eval15_window = (y_eval15_window - y_eval15_window.mean()) / y_eval15_window.std()
# test - normalise
y_test = (y_test - y_test.mean()) / y_test.std()
# Design matrices (created in training.py)
# train
X_train = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\X_train.xlsx')
# test
X_test = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\X_test.xlsx')
# eval15
X_eval15_window = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\X_eval15_window.xlsx')
# eval17
X_eval17_window = pd.read_excel(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\X_eval17_window.xlsx')

# Import the estimated weights
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\W_hat.p", 'rb') as myfile:
    W_hat=cpickle.load(myfile)


# (I) Predict yhat, uhat for each period
# predefine dictionaries for
# yhat
yhat_train = dict()
yhat_test = dict()
yhat_eval15_window = dict()
yhat_eval17_window = dict()
# uhat
uhat_train = dict()
uhat_test = dict()
uhat_eval15_window = dict()
uhat_eval17_window = dict()
# (I.1): Convert the design matrices to numpy array and add constant term
# train
X_train_np = X_train.values
X_train_npb = np.insert(X_train_np, X_train_np.shape[1], 1, axis=1)
# test
X_test_np = X_test.values
X_test_npb = np.insert(X_test_np, X_test_np.shape[1], 1, axis=1)
# eval15
X_eval15_window_np = X_eval15_window.values
X_eval15_window_npb = np.insert(X_eval15_window_np, X_eval15_window_np.shape[1], 1, axis=1)
# eval17
X_eval17_window_np = X_eval17_window.values
X_eval17_window_npb = np.insert(X_eval17_window_np, X_eval17_window_np.shape[1], 1, axis=1)

# (I.2) For all periods loop through the stations and make predictions

# train
for key in y_train:
    # Blank array for actual period, given station
    yhat = np.array([])
    # Loop through time for actual period, given station
    for t in range(0, len(y_train)):
        yhat = np.append(yhat, np.float(NeuralNet(X_train_npb[t, :], W_hat[key], False)))
    yhat_train[key] = yhat
    uhat_train[key] = y_train[key]-yhat
    yhat_train = pd.DataFrame(yhat_train, index=y_train.index)
    uhat_train = pd.DataFrame(uhat_train, index=y_train.index)
# test
for key in y_test:
    # Blank array for actual period, given station
    yhat = np.array([])
    # Loop through time for actual period, given station
    for t in range(0, len(y_test)):
        yhat = np.append(yhat, np.float(NeuralNet(X_test_npb[t, :], W_hat[key], False)))
    yhat_test[key] = yhat
    uhat_test[key] = y_test[key]-yhat
    yhat_test = pd.DataFrame(yhat_test, index=y_test.index)
    uhat_test = pd.DataFrame(uhat_test, index=y_test.index)

# eval15 window
for key in y_eval15_window:
    # Blank array for actual period, given station
    yhat = np.array([])
    # Loop through time for actual period, given station
    for t in range(0, len(y_eval15_window)):
        yhat = np.append(yhat, np.float(NeuralNet(X_eval15_window_npb[t, :], W_hat[key], False)))
    yhat_eval15_window[key] = yhat
    uhat_eval15_window[key] = y_eval15_window[key]-yhat
    yhat_eval15_window = pd.DataFrame(yhat_eval15_window, index=y_eval15_window.index)
    uhat_eval15_window = pd.DataFrame(uhat_eval15_window, index=y_eval15_window.index)

# eval17 window
for key in y_eval17_window:
    # Blank array for actual period, given station
    yhat = np.array([])
    # Loop through time for actual period, given station
    for t in range(0, len(y_eval17_window)):
        yhat = np.append(yhat, np.float(NeuralNet(X_eval17_window_npb[t, :], W_hat[key], False)))
    yhat_eval17_window[key] = yhat
    uhat_eval17_window[key] = y_eval17_window[key]-yhat
    yhat_eval17_window = pd.DataFrame(yhat_eval17_window, index=y_eval17_window.index)
    uhat_eval17_window = pd.DataFrame(uhat_eval17_window, index=y_eval17_window.index)

# (I.3): Vilonplot of residuals
# Rename for plotting
uhat_train_accentname=uhat_train.rename(columns={'Erzsebet': 'Erzsébet',
                                'Korakas': 'Kőrakás',
                                'Kosztolanyi': 'Kosztolányi',
                                'Pesthidegkut': 'Pesthidegkút',
                                'Szena': 'Széna'})
uhat_test_accentname=uhat_test.rename(columns={'Erzsebet': 'Erzsébet',
                                'Korakas': 'Kőrakás',
                                'Kosztolanyi': 'Kosztolányi',
                                'Pesthidegkut': 'Pesthidegkút',
                                'Szena': 'Széna'})
# train
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax=sns.violinplot(data=uhat_train_accentname, inner='box', color='Red', saturation=0.8)
plt.title(r'Violin plot of $\widetilde{PM}_{10}-\widehat{\widetilde{PM}}_{10}$ distributions -- training period')
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(1)[0], figsize(0.8)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\uhat_train_violin.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\uhat_train_violin.png', bbox_inches='tight')
else:
    # show
    plt.show(block=False)
# test
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax=sns.violinplot(data=uhat_test_accentname, inner='box', color='Red', saturation=0.8)
plt.title(r'Violin plot of $\widetilde{PM}_{10}-\widehat{\widetilde{PM}}_{10}$ distributions -- test period')
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(1)[0], figsize(0.8)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\uhat_test_violin.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\uhat_test_violin.png', bbox_inches='tight')
else:
    # show
    plt.show(block=False)

# Make evaluation plots
# Restrict window for plotting
y_eval15_window_r = y_eval15_window['2015-11-05':'2015-11-11']
y_eval17_window_r = y_eval17_window['2017-01-20':'2017-01-29']
yhat_eval15_window_r = yhat_eval15_window['2015-11-05':'2015-11-11']
yhat_eval17_window_r = yhat_eval17_window['2017-01-20':'2017-01-29']
# Plot
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
fig = plt.figure(figsize=(2,4))
# Define the parameter for separating diagonal lines
d = 0.02
# Time array for setting labels
tickarray15 = np.arange('2015-11-05', '2015-11-12', dtype='datetime64[D]')
ticklab15 = tickarray15[[0, 3, 6]]
tickarray17 = np.arange('2017-01-20', '2017-01-30', dtype='datetime64[D]')
ticklab17 = tickarray17[[0, 3, 6, 9]]
myFmt = mdates.DateFormatter('%Y-%m-%d')
# Color settings
truecolor='k'
hatcolor='lime'

# First row: Csepel, Erzsebet
# First row, (1) Csepel
# 2015
ax1 = fig.add_subplot(4, 4, 1)
ax1.plot(y_eval15_window_r['Csepel'], color=truecolor)
ax1.plot(yhat_eval15_window_r['Csepel'], color=hatcolor)
ax1.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax1.set_title('Csepel', fontsize=12)
ax1.set_ylabel(r'$\widetilde{PM}_{10}$', fontsize=12)
# ticks
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.spines['right'].set_visible(False)
plt.setp(ax1.get_yticklabels(), size=10)
kwargs = dict(transform=ax1.transAxes, color='gray', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax2 = fig.add_subplot(4,4,2, sharey = ax1)
ax2.plot(y_eval17_window_r['Csepel'], color=truecolor)
ax2.plot(yhat_eval17_window_r['Csepel'], color=hatcolor)
ax2.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax2.spines['left'].set_visible(False)
ax2.tick_params(labelleft='off')
plt.setp(ax2.get_xticklabels(), visible=False)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)
#First row, (2) Erzsebet
# 2015
ax3 = fig.add_subplot(4,4,3)
ax3.plot(y_eval15_window_r['Erzsebet'], color=truecolor)
ax3.plot(yhat_eval15_window_r['Erzsebet'], color=hatcolor)
ax3.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax3.set_title('Erzsébet', fontsize=12)
# ticks
ax3.spines['right'].set_visible(False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), size=10)
kwargs = dict(transform=ax3.transAxes, color='gray', clip_on=False)
ax3.plot((1-d,1+d), (-d,+d), **kwargs)
ax3.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax4 = fig.add_subplot(4,4,4, sharey = ax3)
ax4.plot(y_eval17_window_r['Erzsebet'], color=truecolor)
ax4.plot(yhat_eval17_window_r['Erzsebet'], color=hatcolor)
ax4.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax4.spines['left'].set_visible(False)
ax4.tick_params(labelleft='off')
plt.setp(ax4.get_xticklabels(), visible=False)
kwargs.update(transform=ax4.transAxes)
ax4.plot((-d,+d), (1-d,1+d), **kwargs)
ax4.plot((-d,+d), (-d,+d), **kwargs)

# Second row: Gergely, Gilice
# Second row, (1) Gergely
# 2015
ax5 = fig.add_subplot(4, 4, 5, sharex=ax1)
ax5.plot(y_eval15_window_r['Gergely'], color=truecolor)
ax5.plot(yhat_eval15_window_r['Gergely'], color=hatcolor)
ax5.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax5.set_title('Gergely', fontsize=12)
ax5.set_ylabel(r'$\widetilde{PM}_{10}$', fontsize=12)
# ticks
ax5.set_xticks(ticklab15)
ax5.xaxis.set_major_formatter(myFmt)
plt.setp(ax5.get_xticklabels(), size=12, visible=False, rotation=45, ha='right')
plt.setp(ax5.get_yticklabels(), size=10)
ax5.spines['right'].set_visible(False)
kwargs = dict(transform=ax5.transAxes, color='gray', clip_on=False)
ax5.plot((1-d,1+d), (-d,+d), **kwargs)
ax5.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax6 = fig.add_subplot(4,4,6, sharex=ax2, sharey = ax1)
ax6.plot(y_eval17_window_r['Gergely'], color=truecolor)
ax6.plot(yhat_eval17_window_r['Gergely'], color=hatcolor)
ax6.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax6.spines['left'].set_visible(False)
ax6.tick_params(labelleft='off')
ax6.set_xticks(ticklab17)
ax6.xaxis.set_major_formatter(myFmt)
plt.setp(ax6.get_xticklabels(), size=12, visible=False, rotation=45, ha='right')
kwargs.update(transform=ax6.transAxes)
ax6.plot((-d,+d), (1-d,1+d), **kwargs)
ax6.plot((-d,+d), (-d,+d), **kwargs)
# Second row, (2) Erzsebet
# 2015
ax7 = fig.add_subplot(4,4,7, sharex=ax3)
ax7.plot(y_eval15_window_r['Gilice'], color=truecolor)
ax7.plot(yhat_eval15_window_r['Gilice'], color=hatcolor)
ax7.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax7.set_title('Gilice', fontsize=12)
# ticks
ax7.spines['right'].set_visible(False)
ax7.set_xticks(ticklab15)
plt.setp(ax7.get_xticklabels(), size=11, visible=False, rotation=45, ha='right')
plt.setp(ax7.get_yticklabels(), size=10)
ax7.xaxis.set_major_formatter(myFmt)
kwargs = dict(transform=ax7.transAxes, color='gray', clip_on=False)
ax7.plot((1-d,1+d), (-d,+d), **kwargs)
ax7.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax8 = fig.add_subplot(4,4,8, sharex=ax4, sharey = ax3)
ax8.plot(y_eval17_window_r['Gilice'], color=truecolor)
ax8.plot(yhat_eval17_window_r['Gilice'], color=hatcolor)
ax8.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax8.spines['left'].set_visible(False)
ax8.tick_params(labelleft='off')
ax8.set_xticks(ticklab17)
ax8.xaxis.set_major_formatter(myFmt)
plt.setp(ax8.get_xticklabels(), size=11, visible=False, rotation=45, ha='right')
kwargs.update(transform=ax8.transAxes)
ax8.plot((-d,+d), (1-d,1+d), **kwargs)
ax8.plot((-d,+d), (-d,+d), **kwargs)

# Third row: Korakas, Kosztolanyi
# Third row, (1) Korakas
# 2015
ax9 = fig.add_subplot(4, 4, 9, sharex=ax5)
ax9.plot(y_eval15_window_r['Korakas'], color=truecolor)
ax9.plot(yhat_eval15_window_r['Korakas'], color=hatcolor)
ax9.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax9.set_title('Korakas', fontsize=12)
ax9.set_ylabel(r'$\widetilde{PM}_{10}$', fontsize=12)
# ticks
plt.setp(ax9.get_xticklabels(), visible=False)
ax9.spines['right'].set_visible(False)
plt.setp(ax9.get_yticklabels(), size=10)
kwargs = dict(transform=ax9.transAxes, color='gray', clip_on=False)
ax9.plot((1-d,1+d), (-d,+d), **kwargs)
ax9.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax10 = fig.add_subplot(4,4,10, sharex=ax6, sharey = ax9)
ax10.plot(y_eval17_window_r['Korakas'], color=truecolor)
ax10.plot(yhat_eval17_window_r['Korakas'], color=hatcolor)
ax10.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax10.spines['left'].set_visible(False)
ax10.tick_params(labelleft='off')
plt.setp(ax10.get_xticklabels(), visible=False)
kwargs.update(transform=ax10.transAxes)
ax10.plot((-d,+d), (1-d,1+d), **kwargs)
ax10.plot((-d,+d), (-d,+d), **kwargs)
#Thrid row, (2) Kosztolanyi
# 2015
ax11 = fig.add_subplot(4,4,11, sharex=ax7)
ax11.plot(y_eval15_window_r['Kosztolanyi'], color=truecolor)
ax11.plot(yhat_eval15_window_r['Kosztolanyi'], color=hatcolor)
ax11.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax11.set_title('Kosztolányi', fontsize=12)
# ticks
ax11.spines['right'].set_visible(False)
plt.setp(ax11.get_xticklabels(), visible=False)
plt.setp(ax11.get_yticklabels(), size=10)
kwargs = dict(transform=ax11.transAxes, color='gray', clip_on=False)
ax11.plot((1-d,1+d), (-d,+d), **kwargs)
ax11.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax12 = fig.add_subplot(4,4,12, sharex=ax8, sharey = ax11)
ax12.plot(y_eval17_window_r['Kosztolanyi'], color=truecolor)
ax12.plot(yhat_eval17_window_r['Kosztolanyi'], color=hatcolor)
ax12.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax12.spines['left'].set_visible(False)
ax12.tick_params(labelleft='off')
plt.setp(ax12.get_xticklabels(), visible=False)
kwargs.update(transform=ax12.transAxes)
ax12.plot((-d,+d), (1-d,1+d), **kwargs)
ax12.plot((-d,+d), (-d,+d), **kwargs)

# Fourth row: Pesthidegkut, Szena
# Fourth row, (1) Pesthidegkut
# 2015
ax13 = fig.add_subplot(4, 4, 13, sharex=ax9)
ax13.plot(y_eval15_window_r['Pesthidegkut'], color=truecolor)
ax13.plot(yhat_eval15_window_r['Pesthidegkut'], color=hatcolor)
ax13.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax13.set_title('Pesthidegkút', fontsize=12)
ax13.set_ylabel(r'$\widetilde{PM}_{10}$', fontsize=12)
# ticks
ax13.set_xticks(ticklab15)
ax13.xaxis.set_major_formatter(myFmt)
plt.setp(ax13.get_xticklabels(), size=10, visible=True, rotation=30, ha='right')
plt.setp(ax13.get_yticklabels(), size=10)
ax13.spines['right'].set_visible(False)
kwargs = dict(transform=ax13.transAxes, color='gray', clip_on=False)
ax13.plot((1-d,1+d), (-d,+d), **kwargs)
ax13.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax14 = fig.add_subplot(4,4,14, sharex=ax10, sharey = ax13)
ax14.plot(y_eval17_window_r['Pesthidegkut'], color=truecolor)
ax14.plot(yhat_eval17_window_r['Pesthidegkut'], color=hatcolor)
ax14.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax14.spines['left'].set_visible(False)
ax14.tick_params(labelleft='off')
ax14.set_xticks(ticklab17)
ax14.xaxis.set_major_formatter(myFmt)
plt.setp(ax14.get_xticklabels(), size=10, visible=True, rotation=30, ha='right')
kwargs.update(transform=ax14.transAxes)
ax14.plot((-d,+d), (1-d,1+d), **kwargs)
ax14.plot((-d,+d), (-d,+d), **kwargs)
# Fourth row, (2) Szena
# 2015
ax15 = fig.add_subplot(4,4,15, sharex=ax11)
ax15.plot(y_eval15_window_r['Szena'], color=truecolor)
ax15.plot(yhat_eval15_window_r['Szena'], color=hatcolor)
ax15.axvline(x=mdates.datestr2num('2015-11-08'), linewidth=5, color='red', alpha=0.5)
ax15.set_title('Széna', fontsize=12)
# ticks
ax15.spines['right'].set_visible(False)
ax15.set_xticks(ticklab15)
plt.setp(ax15.get_xticklabels(), size=10, visible=True, rotation=30, ha='right')
plt.setp(ax15.get_yticklabels(), size=10)
ax15.xaxis.set_major_formatter(myFmt)
kwargs = dict(transform=ax15.transAxes, color='gray', clip_on=False)
ax15.plot((1-d,1+d), (-d,+d), **kwargs)
ax15.plot((1-d,1+d),(1-d,1+d), **kwargs)
# 2017
ax16 = fig.add_subplot(4,4,16, sharex=ax12, sharey = ax15)
ax16.plot(y_eval17_window_r['Szena'], color=truecolor)
ax16.plot(yhat_eval17_window_r['Szena'], color=hatcolor)
ax16.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.5)
# ticks
ax16.spines['left'].set_visible(False)
ax16.tick_params(labelleft='off')
ax16.set_xticks(ticklab17)
ax16.xaxis.set_major_formatter(myFmt)
plt.setp(ax16.get_xticklabels(), size=10, visible=True, rotation=30, ha='right')
kwargs.update(transform=ax16.transAxes)
ax16.plot((-d,+d), (1-d,1+d), **kwargs)
ax16.plot((-d,+d), (-d,+d), **kwargs)

# Adjust subplots
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(hspace=0.8)
# Save or view
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(1)[0], figsize(1.3)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\evaluation.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\evaluation.png', bbox_inches='tight')
else:
    # show
    plt.show(block=True)