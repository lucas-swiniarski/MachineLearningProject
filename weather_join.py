import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
import time
import csv

#path = "../../../Google Drive/Data_science/NYU/Machine Learning/ML Project (Collisions)/" #Joe
path = "../../../../Google Drive/ML Project (Collisions)/" # Joyce
# path = "" # Lucas

#Read in the collision data
df = pd.read_csv(path + "NYPD_Motor_Vehicle_Collisions.csv", parse_dates=[['DATE', 'TIME']], infer_datetime_format=True)
df = df[df.LATITUDE.notnull()]
df['LOCATION'] = [(float(x), (y)) for x,y in zip(df['LATITUDE'], df['LONGITUDE'])]

#Read in the stations data
stations = pd.read_csv(path + "wunderground_stations.csv")
stations['LOCATION'] = [(x, y) for x,y in zip(stations['lat'], stations['long'])]
stations = stations[stations['station'] != 'KJRB']

points = list(stations['LOCATION'])

def closest_station(point):
    """ Find closest point from a list of points. """
    minpt = cdist([point], points).argmin()
    return list(stations['station'])[minpt]

df['station'] = [closest_station(x) for x in df.LOCATION]

weather = pd.read_csv(path + "wunderground_weather.tsv", delimiter='\t', parse_dates=[['date', 'time']], infer_datetime_format=True)


def closest_time(station, time):
    search = weather[weather['station']==station]['date_time'].searchsorted(time)[0]
    index = weather[weather['station']==station].iloc[search].name
    return index

weather_index = []

i = 0
start_time = time.time()

for x,y in zip(df['station'], df['DATE_TIME']):
    weather_index.append(closest_time(x, y))
    i += 1
    
    if i % 100 == 0:
        elapsed_time = time.time() - start_time
        print '%s rows completed, approximate time remaining: %s' % (i, elapsed_time/float(i)*(792851 - i))

with open('weather_joined.pkl', 'wb') as output:
    pickle.dump(weather_index, output, pickle.HIGHEST_PROTOCOL)

