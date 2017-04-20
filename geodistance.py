# (I): Create a dictionary of coordinates of all stations
# (II): Computing the distances between stations and export into excel file, print LaTeX table string. Also compute
#       the distance of stations from the meteorological station, print to LaTeX and excel
# (III.1): Create a function which calculates inverse-distance based weights for stations based on coordinates
# (III.2): Create a parsimonious version of (III.1) based on distances

#Importing dependencies
import pandas as pd
import csv

if __name__=='__main__':
    #Reading stations
    #all stations
    with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\All_Stations.csv", 'r') as myfile:
        rd=csv.reader(myfile, delimiter=',')
        #list the headers
        allstations=list(rd)
        #correct for the returned list of list file only to have a simple a list
        allstations=allstations[0]
    #  kept stations
    with open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Kept_Stations.csv", 'r') as myfile:
        rd=csv.reader(myfile, delimiter=',')
        #list the headers
        keptstations=list(rd)
        #correct for the returned list of list file only to have a simple a list
        keptstations=keptstations[0]
    print('The kept stations are \n', keptstations, '\n\n')

    #(I): Create a DataFrame containing all the stations and the coordinates in decimal degrees, print to LaTex
    # and export to excel
    #DataFrame
    d={'Teteny':[47.4061243, 19.0091796],'Csepel':[47.4047352, 19.0911827], 'Erzsebet':[47.4975260, 19.0527480],
     'Gergely':[47.4674130, 19.1558940], 'Gilice':[47.4309037, 19.1811572], 'Honved':[47.5217225, 19.0684197],
     'Kaposztas':[47.5853036, 19.1147661], 'Korakas':[47.5433350,19.1465781], 'Kosztolanyi':[47.4759677, 19.0400691],
    'Pesthidegkut':[47.5620970, 18.9614345], 'Szena':[47.5078719, 19.0267938], 'Teleki':[47.4934215, 19.0847853]}
    coordinates=pd.DataFrame(d, index=['latitude', 'longitude'])
    d={'Teteny':[47.4061243, 19.0091796],'Csepel':[47.4047352, 19.0911827], 'Erzsebet':[47.4975260, 19.0527480],
     'Gergely':[47.4674130, 19.1558940], 'Gilice':[47.4309037, 19.1811572], 'Honved':[47.5217225, 19.0684197],
     'Kaposztas':[47.5853036, 19.1147661], 'Korakas':[47.5433350,19.1465781], 'Kosztolanyi':[47.4759677, 19.0400691],
    'Pesthidegkut':[47.5620970, 18.9614345], 'Szena':[47.5078719, 19.0267938], 'Teleki':[47.4934215, 19.0847853],
       'Csillaghegy_met': [47.5957565308, 19.0407772064]}
    coordinates_extended=pd.DataFrame(d, index=['latitude', 'longitude'])
    #print to LaTeX
    print('###### LaTeX table for the coordinates of stations\n', coordinates.T.to_latex(),'\n ########\n')
    #export to excel
    coordinates.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Coordinates.xlsx")
    print('The coordinates of all PM10 stations are\n', coordinates, '\n\n')



#(II) Distance matrix
#(II.1) Haversine function
#(II.2) Distance matrix: function taking the DataFrame of all coordinates and the name of the station
#     and returns its distance from every station as an array
#(II.3) Distance matrix: call '(II.2)' for every station to make a dictionary, conversion to DataFrame, export to excel,
#     print LaTeX table string
#(II.4) Distance matrix extended version: with meteorological station included, export to excel, print to LaTeX

#(II.1) Haversine, credits to Michael Dunn, Uppsala University,
# posted http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing
# -and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    #import math dependencies
    from math import radians, cos, sin, asin, sqrt
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

#(II.2) distance function
def distancearray(df,s):
    """
    Accepts  (1) a DataFrame object (df) which contains locations as columns and the rows
    are indexed by 'latitude' and 'longitude' corresponding to coordinates of the
    locations in the columns in decimal degrees;
    (2) the name of the column (s) for which the distances from the other locations are
     to be determined.
     Returns an array of distances (dist) in kilometers, where the first element corresponds
     to the first location in the DataFrame.
     Dependencies: pandas, Haversine
    :param s:
    :return:
    """
    #empty array to be filled
    dist=list()
    #looping through all the locations in columns and determine the distance from
    #the location of interest, df['s']
    for key in df:
        dist.append(haversine(df.ix['longitude'][s], df.ix['latitude'][s],
                                   df.ix['longitude'][key],df.ix['latitude'][key]))
    #return the array
    return dist

if __name__=='__main__':
    #(II.3)-(II.4) call 'distancearray' function for every station to create a dictionary, convert to DataFrame,
    # export to excel, print LaTeX table string code
    #distancearray function
    d1={key: distancearray(coordinates,key) for key in coordinates}
    d2={key: distancearray(coordinates_extended,key) for key in coordinates_extended}
    #convert the dictionary to DataFrame
    distancematrix=pd.DataFrame(d1,index=coordinates.columns)
    distancematrix_extended=pd.DataFrame(d2, index=coordinates_extended.columns)
    #export to excel
    distancematrix.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Distancematrix.xlsx")
    distancematrix_extended.to_excel(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Data\Distancematrix_extend.xlsx")
    #print to LaTeX
    #rounding to 4 decimals
    def rounding(x):
        return x.round(decimals=4)
    #printing
    print('######## \n\n LaTeX table for distancematrix \n',
       distancematrix.apply(rounding).to_latex(), '\n ########\n')
    print('######## \n\n LaTeX table for extended distancematrix \n',
       distancematrix_extended.apply(rounding).to_latex(), '\n ########\n')

#(III.1) InvDistWeights: function taking the DataFrame of all coordinates, the name of the station,
#      the list of kept stations, and returns the inverse-distance weights, based on
#       station's distance from all the kept stations, as an array
# InvDistWeights
def InvDistWeights(df_coord,s,keptlist):
    """
    Accepts  (1) a DataFrame object (df) which contains locations as columns and the rows
    are indexed by 'latitude' and 'longitude' corresponding to coordinates of the
    locations in the columns in decimal degrees;
    (2) the name of the column (s) for which the distances from the other locations are
     to be determined
    (3) a list of kept locations for which the inverse-distance weights are calculated for
     the station (s)
     Returns an array of inverse-distance weights (weights), where the first element corresponds
     to the first kept location in the DataFrame, the array adds up to 1
     Dependencies: pandas, Haversine
    :param df_coord: DataFrame containing the locations as columns and the rows are indexed by 'latitude' and 'longitude'
            corresponding to coordinates of the locations in the columns
    :param s: name of the location for which the weights are to be determined
    :param keptlist: a list of kept locations for which the inverse-distance weights are calculated for
            the station (s)
    :return: array of the weights, of length same as 'keptlist'
    """
    #Distance from locations in 'keptlist'
    #empty array to be filled
    dist=list()
    # looping through all the locations in columns of DataFrame which is also in 'keptlist'
    #  and determine the distance from the location of interest, df['s']
    for key in df_coord:
         if key in keptlist:
             dist.append(haversine(df.ix['longitude'][s], df.ix['latitude'][s],
                                   df.ix['longitude'][key], df.ix['latitude'][key]))
    #Weights:
    #dist2weights function
    def dist2weights(dlist):
        #reciprocal
        def reciprocal(x):
            if x==0:
                return x
            else:
                return 1/x
        out=list(map(reciprocal, dlist))
        out=list(map(lambda x: x/sum(out), out))
        return out
    return dist2weights(dist)

#(III.2) Parsimonius InvDistWeights function (doesn't calculate the distances at every single call, but relies on
#the DataFrame of distances)
def InvDistWeightsP(df_dist,s,keptlist):
    """
    Returns inverse-distance weights for a location from a distance matrix, based on the distance of the given location
    from selected locations (contributors to weighting)
    (Parsimonious version of InvDistWeights)
    :param df_dist: DataFrame of distance matrix between locations
    :param s:  the location in DataFrame for which the weights are required
    :param keptlist: the stations that are used as weighting ones (contributors)
    :return: an array of weights, of length same as 'keptlist'
    """
    #Extract the appropriate distances from the appropriate column
    dist=list()
    for idx in keptlist:
        dist.append(df_dist.ix[idx, s])
    #Turn the selected distances to weights
    #dist2weights function
    def dist2weights(dlist):
        #reciprocal
        def reciprocal(x):
            if x==0:
                return x
            else:
                return 1/x
        out=list(map(reciprocal, dlist))
        out=list(map(lambda x: x/sum(out), out))
        return out
    return dist2weights(dist)
