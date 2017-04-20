# Importing the data:
# (I): the PM10 data from Excel file, converting to numeric, drop stations,
#   sorting data by stations, export data to excel
# (II): the meteorological data from csv file, converting to numeric, export to excel (note: January 2017 data has been
#       added previously manually to the files from 'Obuda_2017januar.ods'; strings indicating temparature measurement
#        units differ from that of in 2013-2016)



#Loading dependencies
import pandas as pd
from datetime import datetime
import csv
#in addition, 'openpyxl' is needed to write into excel


#(I) PM10 data

# Import the file
path=r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Smog_PM10.xlsx"
y=pd.read_excel(path, sheetname=['2013','2014','2015','2016','2017'], na_values=['Nincs adat'])


# Rename stations
for key in y:
    y[key]=y[key].rename(columns={'Dátum': 'dates',
                        'Budapest Budatétény': 'Teteny',
                        'Budapest Csepel': 'Csepel',
                        'Budapest Erzsébet tér': 'Erzsebet',
                        'Budapest Gergely utca': 'Gergely',
                        'Budapest Gilice tér': 'Gilice',
                        'Budapest Honvéd': 'Honved',
                        'Budapest Káposztásmegyer': 'Kaposztas',
                        'Budapest Korakás park': 'Korakas',
                        'Budapest Kosztolányi D. tér': 'Kosztolanyi',
                        'Budapest Pesthidegkút': 'Pesthidegkut',
                        'Budapest Széna tér': 'Szena',
                        'Budapest Teleki tér': 'Teleki'})

#Write the renamed stations (all) name into .csv file for geodistance
allstations=list(y['2013'])
#deleting 'date' column
allstations=allstations[1:]
#write
path=r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\All_Stations.csv"
with open(path, 'w', newline='') as myfile:
    wr=csv.writer(myfile, delimiter=',')
    wr.writerow(allstations)


#Dropping stations
for key in y:
    for todelete in ('Teteny', 'Kaposztas', 'Teleki', 'Honved'):
       y[key]=y[key].drop(todelete, axis=1)

#Write the renamed not dropped stations name into .csv for geodistance
keptstations=list(y['2013'])
#deleting 'date' column
keptstations=keptstations[1:]
#write
path=r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Kept_Stations.csv"
with open(path, 'w', newline='') as myfile:
    wr=csv.writer(myfile, delimiter=',')
    wr.writerow(keptstations)


#Setting date and sort by it
for key in y:
    #Convert to date
    y[key]['dates']=pd.to_datetime(y[key]['dates'])
    #Sort by it
    y[key]=y[key].sort_values(by='dates')
    #Set as index
    y[key].index=y[key]['dates']
    #Delete Date column
    del y[key]['dates']



#Cutting the ug/m^3 tag and converting the data to numbers

#cutting white spaces
for key in y:
    for key1 in y[key]:
        pd.core.strings.str_strip(y[key][key1])

#cutting the ug/m^3 tag for not NaNs
#create a cutting function which will be applied to the rows
def cuttags(s):
    if pd.notnull(s):
         return s[:len(s)-5]

#apply the cutting and convert to numeric
for key in y:
    y[key]=y[key].applymap(func=cuttags)
    y[key]=y[key].apply(func=pd.to_numeric)


#Stacking the data, sorting by stations
y_s=y['2013']
for key in ('2014', '2015', '2016','2017'):
    y_s=y_s.append(y[key])

#Save the file
y_s.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Smog_PM10_stacked.xlsx", na_rep='NaN')

#(II) Meteorological data

#Importing the files, index by date
#Import average temperature
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\67_avghom.csv", 'r') as myfile:
    temp_avg=pd.read_csv(myfile, sep=';', header=0)
#Import minimum temperature
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\67_minhom.csv", 'r') as myfile:
    temp_min=pd.read_csv(myfile, sep=';', header=0)
#Import maximum temperature
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\67_maxhom.csv", 'r') as myfile:
    temp_max=pd.read_csv(myfile, sep=';', header=0)
#Import windpower
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\67_szelero.csv", 'r') as myfile:
    wind_power=pd.read_csv(myfile, sep=';', header=0)
#Import windblow
with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\67_szellokes.csv",'r') as myfile:
    wind_blow=pd.read_csv(myfile, sep=';', header=0)


#Rename columns
temp_avg=temp_avg.rename(columns={'HelyszĂ­n': 'location', 'DĂˇtum': 'dates', 'hom': 'temp_avg'})
temp_min=temp_min.rename(columns={'HelyszĂ­n': 'location', 'DĂˇtum': 'dates', 'min_hom': 'temp_min'})
temp_max=temp_max.rename(columns={'HelyszĂ­n': 'location', 'DĂˇtum': 'dates', 'max_hom': 'temp_max'})
wind_power=wind_power.rename(columns={'HelyszĂ­n': 'location', 'DĂˇtum': 'dates', 'szelero': 'wind_power'})
wind_blow=wind_blow.rename(columns={'HelyszĂ­n': 'location', 'DĂˇtum': 'dates', 'szellokes': 'wind_blow'})

#Setting the date, sort by it
#list of DatFrame
metlist=[temp_avg, temp_min, temp_max, wind_power, wind_blow]
# Setting date and sort by it
for x in range(0,len(metlist)):
    #convert to date
    metlist[x]['dates'] = pd.to_datetime(metlist[x]['dates'])
    #sort by it
    metlist[x]=metlist[x].sort_values(by='dates')
    #set as index
    metlist[x].index = metlist[x]['dates']
    #delete dates  and locationcolumn
    del metlist[x]['dates']
    del metlist[x]['location']
#loading back the data from the list
temp_avg, temp_min, temp_max, wind_power, wind_blow=metlist

#Cutting the measurement unit strings and convert into numbers
#cut white spaces
for x in range(0,len(metlist)):
    for key in metlist[x]:
        pd.core.strings.str_strip(metlist[x][key])
# loading back the data from the list
temp_avg, temp_min, temp_max, wind_power, wind_blow = metlist
#create a function which cuts 'Â°C' tags; and '°C' tags from 2017 January
def cuttagsTemp(s,isfromjanuary17):
    if (pd.notnull(s) and isfromjanuary17==False):
         return s[:len(s)-3]
    if (pd.notnull(s) and isfromjanuary17==True):
        return s[:len(s)-2]

def cuttagsTemp_notjanuary17(s):
    return cuttagsTemp(s,False)

def cuttagsTemp_january17(s):
    return cuttagsTemp(s,True)

#create a function which cuts 'km/h' tags
def cuttagsWind(s):
    if pd.notnull(s):
         return s[:len(s)-4]

#apply the cuttagsTemp to temperatures
temp_avg[:'2016-12-31']=temp_avg[:'2016-12-31'].applymap(func=cuttagsTemp_notjanuary17)
temp_avg['2017-01-01':]=temp_avg['2017-01-01':].applymap(func=cuttagsTemp_january17)
temp_min[:'2016-12-31']=temp_min[:'2016-12-31'].applymap(func=cuttagsTemp_notjanuary17)
temp_min['2017-01-01':]=temp_min['2017-01-01':].applymap(func=cuttagsTemp_january17)
temp_max[:'2016-12-31']=temp_max[:'2016-12-31'].applymap(func=cuttagsTemp_notjanuary17)
temp_max['2017-01-01':]=temp_max['2017-01-01':].applymap(func=cuttagsTemp_january17)
#apply the cuttagsWind for winds
wind_blow=wind_blow.applymap(func=cuttagsWind)
wind_power=wind_power.applymap(func=cuttagsWind)
#create a function which takes numbers in string form with comma as decimal separator
def commadecimal2numeric(s):
    return pd.to_numeric(s.replace(',','.'))
#convert all values to numeric
temp_avg=temp_avg.applymap(func=commadecimal2numeric)
temp_min=temp_min.applymap(func=commadecimal2numeric)
temp_max=temp_max.applymap(func=commadecimal2numeric)
wind_blow=wind_blow.applymap(func=commadecimal2numeric)
wind_power=wind_power.applymap(func=commadecimal2numeric)

#Write to excel
temp_avg.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_avg.xlsx")
temp_min.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_min.xlsx")
temp_max.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\temp_max.xlsx")
wind_power.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\wind_power.xlsx")
wind_blow.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\wind_blow.xlsx")