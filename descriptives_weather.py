#For meteorological data
# (I): Find missing dates, produce and check descriptives, fix anomalies
# (II): Impute missing dates
# (III): Provide corrected descriptives, print to LaTeX
# (IV): Export the corrected, imputed data to excel
# (V): Plot the data

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


#(I): Finding missing dates, produce and check descriptives, fix anomalies

#Importing
temp_avg=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_avg.xlsx")
temp_avg.name='temp_avg'
temp_min=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_min.xlsx")
temp_min.name='temp_min'
temp_max=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_max.xlsx")
temp_max.name='temp_max'
wind_power=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\wind_power.xlsx")
wind_power.name='wind_power'
wind_blow=pd.read_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\wind_blow.xlsx")
wind_blow.name='wind_blow'
#list
metlist=[temp_avg, temp_min, temp_max, wind_power, wind_blow]

#Finding out the missing dates
missing=dict()
missing_number=dict()
for x in metlist:
    #create the whole range of dates in the time window
    date_set=set(x['dates'][0]+dt.timedelta(c)
              for c in range((x['dates'][len(x)-1]-x['dates'][0]).days))
    #find the setminus with the dates in the data
    missing[x.name]=(date_set-set(x['dates']))
    missing_number[x.name]=len(missing[x.name])
print('The number of missing dates/values in the meteorological data is given by \n', missing_number, '\n\n')

#Produce and check descriptives
descr_met=dict()
for x in metlist:
    #set the index to the dates
    x.index=x['dates']
    #delete 'dates' column
    del x['dates']
    #produce descriptives
    descr_met[x.name+'_descr']=pd.DataFrame({key: [np.mean(x[key]),
                                             x.median()[key],
                                             stats.mode(x[key])[0][0],
                                             np.max(x[key]),
                                             np.min(x[key]),
                                             np.std(x[key]),
                                            missing_number[x.name],
                                             len(x[key])] for key in x},
                                      index=['Mean', 'Median' ,'Mode', 'Max', 'Min' , 'Std',
                                             'No. of missing observations',
                                             'No. of existing observations'])
descr_df=pd.concat(descr_met,axis=1)
print('Preliminary descriptives to check:', descr_df, '\n\n')

#Fix anomalies: max wind blow is 1908 km/h, substitute with mean ('window_blow' is the 4th element in metlist)
metlist[4]['wind_blow'][metlist[4]['wind_blow']==1908]=np.median(metlist[4]['wind_blow'])


# (II): Impute the missing value with all-time average
#new list
metlist1=list()
for x in metlist:
    #create a DataFrame indexed by the missing dates
    df_mean=pd.DataFrame({x.name: np.mean(x[x.name])}, index=missing[x.name])
    #concatinate with the original
    x_imp=pd.concat([x, df_mean])
    x_imp.name=x.name
    #sort by dates
    x_imp.sort_index(inplace=True)
    #add DataFrame to new list
    metlist1.append(x_imp)
#loading back
temp_avg, temp_min, temp_max, wind_power, wind_blow=metlist1

# (III): Provide descriptives for the corrected, imputed data, print to LaTeX
descr_met_corr=dict()
for x in metlist1:
    descr_met_corr[x.name+'_descr']=pd.DataFrame({key: [np.mean(x[key]),
                                             x.median()[key],
                                             stats.mode(x[key])[0][0],
                                             np.max(x[key]),
                                             np.min(x[key]),
                                             np.std(x[key]),
                                            missing_number[x.name],
                                             len(x[key])] for key in x},
                                      index=['Mean', 'Median' ,'Mode', 'Max', 'Min' , 'Std',
                                             'No. of imputed observations',
                                             'No. of days in the period'])
descr_corr_df=pd.concat(descr_met_corr,axis=1)

#Printing to LaTeX table
#rounding function
def rounding(x):
    return x.round(decimals=4)
print('######## \n\n LaTeX table for the corrected, imputed descriptives of meteo data \n',
      descr_corr_df.apply(rounding).to_latex(), '\n ########\n')

#(IV): Export the corrected data to excel
for x in metlist1:
    x.to_excel('C:\\Users\\Máté\\Dropbox\\CEU\\2017 Winter\\Econ\\Smog\\Data\\'+x.name+'_imputed.xlsx')


# (V): Plot the corrected, imputed meteorological data
# Temperature
plt.clf()
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
fig = plt.figure()
ax = fig.add_subplot(111)  # big subplot
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.grid(False)
# Common label
ax.set_ylabel('Temperature ($^{\circ}C$)', size=12)
ax.set_title('Time series of Temperature -- the whole and Evaluation 2 period')
# Left subplot
ax1.plot(temp_max, label='Max. temperature', color='red', linewidth=0.9)
ax1.plot(temp_avg, label='Avg. temperature', color='k', linewidth=0.9)
ax1.plot(temp_min, label='Max. temperature', color='cornflowerblue', linewidth=0.9)
ax1.legend()
plt.setp(ax1.get_xticklabels(), size=12, visible=True, rotation=45, ha='right')
ax1.grid(False)
# Right subplot
ax2.plot(temp_max['2017-01-18':'2017-01-28'], color='red')
ax2.plot(temp_avg['2017-01-18':'2017-01-28'], color='k')
ax2.plot(temp_min['2017-01-18':'2017-01-28'], color='cornflowerblue')
ax2.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.65)
ax2.text(*mdates.datestr2num(['2017-01-19']), 3, 'Eval. 2', size=12, color='w',
         bbox={'facecolor': 'red', 'edgecolor': 'gray', 'alpha': 0.65, 'pad': 1.5})
# Setting ticks
a=np.arange('2017-01-18', '2017-01-29', dtype='datetime64[D]')
ax2.set_xticks(a[[0, 5, 8, 10]])
myFmt = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(myFmt)
plt.setp(ax2.get_xticklabels(), size=12, visible=True, rotation=45, ha='right')
ax2.grid(False)
plt.tight_layout()
if savefigure_mode == True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(0.9)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\temperature.pgf',
                bbox_inches='tight')  # , dip=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\temperature.png', bbox_inches='tight')  # , dip=500
else:
    # show
    plt.show(block=True)


# Wind blow
plt.clf()
plt.rc('font',family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
fig = plt.figure()
ax = fig.add_subplot(111)    # The big subplot
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.grid(False)
# Common label
ax.set_ylabel('Wind blow ($kmh^{-1}$)', size=12)
ax.set_title('Time series of Wind blow -- the whole and Evaluation 2 period')
# Left subplot
ax1.plot(wind_blow, label='Wind blow', linewidth=0.9, color='cornflowerblue')
ax1.legend()
plt.setp(ax1.get_xticklabels(), size=12, visible=True, rotation=45, ha='right')
ax1.grid(False)
# Right subplot
ax2.plot(wind_blow['2017-01-18':'2017-01-28'], color='cornflowerblue')
ax2.axvspan(*mdates.datestr2num(['2017-01-23', '2017-01-26']), color='red', alpha=0.65)
ax2.text(*mdates.datestr2num(['2017-01-19']), 29, 'Eval. 2', size=12, color='w',
        bbox={'facecolor':'red', 'edgecolor': 'gray', 'alpha':0.65, 'pad':1.5})
# Setting ticks
a=np.arange('2017-01-18', '2017-01-29', dtype='datetime64[D]')
ax2.set_xticks(a[[0, 5, 8, 10]])
myFmt = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(myFmt)
plt.setp(ax2.get_xticklabels(), size=12, visible=True, rotation=45, ha='right')
ax2.grid(False)
if savefigure_mode==True:
    fig = plt.gcf()
    fig.set_size_inches(figsize(1)[0], figsize(0.8)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\windblow.pgf', bbox_inches='tight')#, dip=500)
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\latex\windblow.png', bbox_inches='tight')#, dip=500
else:
    # show
    plt.show(block=True)