import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
import time
import csv

import math

class OutOfRangeError(ValueError):
    pass

__all__ = ['to_latlon', 'from_latlon']

K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = math.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None):

    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if not 100000 <= easting < 1000000:
        raise OutOfRangeError('easting out of range (must be between 100.000 m and 999.999 m)')
    if not 0 <= northing <= 10000000:
        raise OutOfRangeError('northing out of range (must be between 0 m and 10.000.000 m)')
    if not 1 <= zone_number <= 60:
        raise OutOfRangeError('zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise OutOfRangeError('zone letter out of range (must be between C and X)')

        northern = (zone_letter >= 'N')

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu +
             P2 * math.sin(2 * mu) +
             P3 * math.sin(4 * mu) +
             P4 * math.sin(6 * mu) +
             P5 * math.sin(8 * mu))

    p_sin = math.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = math.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = math.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = _E * p_cos**2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) *
                (d2 / 2 -
                 d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                 d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d -
                 d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    return (math.degrees(latitude),
            math.degrees(longitude) + zone_number_to_central_longitude(zone_number))


def from_latlon(latitude, longitude, force_zone_number=None):
    if not -80.0 <= latitude <= 84.0:
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not -180.0 <= longitude <= 180.0:
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')

    lat_rad = math.radians(latitude)
    lat_sin = math.sin(lat_rad)
    lat_cos = math.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    zone_letter = latitude_to_zone_letter(latitude)

    lon_rad = math.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = math.radians(central_lon)

    n = R / math.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * math.sin(2 * lat_rad) +
             M3 * math.sin(4 * lat_rad) -
             M4 * math.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if latitude < 0:
        northing += 10000000

    return easting, northing


def latitude_to_zone_letter(latitude):
    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def latlon_to_zone_number(latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180) / 6) + 1


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3

def get_utm(latitude, longitude):
    utm1, utm2 = from_latlon(latitude, longitude)
    return (utm1, utm2)

#path = "../../../Google Drive/Data_science/NYU/Machine Learning/ML Project (Collisions)/" #Joe
path = "../../../../Google Drive/ML Project (Collisions)/" # Joyce
# path = "" # Lucas

#Read in the collision data
df = pd.read_csv(path + "NYPD_Motor_Vehicle_Collisions.csv", parse_dates=[['DATE', 'TIME']], infer_datetime_format=True)
df.LATITUDE = df.LATITUDE.astype(float)
df.LONGITUDE = df.LONGITUDE.astype(float)
df = df[df.LATITUDE.notnull()]
df = df[df.LONGITUDE > -100]
df['LOCATION'] = [get_utm(x,y) for x,y in zip(df['LATITUDE'], df['LONGITUDE'])]

#Read in the stations data
stations = pd.read_csv(path + "wunderground_stations.csv")
stations['lat']=stations['lat'].astype(float)
stations['long']=stations['long'].astype(float)
stations['LOCATION'] = [get_utm(x,y) for x,y in zip(stations['lat'], stations['long'])]
stations = stations[stations['station'] != 'KJRB']

points = list(stations['LOCATION'])

def closest_station(point):
    """ Find closest point from a list of points. """
    minpt = cdist([point], points).argmin()
    return list(stations['station'])[minpt]

df['station'] = [closest_station(x) for x in df.LOCATION]

print 'Finished searching for closest station'

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

#Left join on weather data
df['weather_join_index'] = weather_index
df = df.drop(['station'], axis=1).join(weather, on='weather_join_index',  how='left')
df = df.set_index(['UNIQUE KEY'])

#Remove columns that are not needed for the join
df = df.loc[:,['temperature', 'heat_index', 'dew_point',
       'humidity', 'pressure', 'visibility', 'wind_dir', 'wind_speed',
       'gust_speed', 'precip', 'events', 'conditions']]

#Data cleaning and transformation
df['Rain'] = ['Rain' in i for i in df['events']]
df['Thunderstorm'] = ['Thunderstorm' in i for i in df['events']]
df['Snow'] = ['Snow' in i for i in df['events']]
df['Fog'] = ['Fog' in i for i in df['events']]
df['temperature'] = [i.strip(' F') for i in df.temperature]
df['heat_index'] = [i.strip(' F') for i in df.heat_index]
df['dew_point'] = [i.strip(' F') for i in df.dew_point]
df['humidity'] = [i.strip('%') for i in df.humidity]
df['pressure'] = [i.strip(' in') for i in df.humidity]
df['visibility'] = [i.strip(' mi') for i in df.visibility]
df['wind_speed'] = [i.strip(' mph') for i in df.wind_speed]
df['gust_speed'] = [i.strip(' mph') for i in df.gust_speed]
df['precip'] = [str(i).strip(' in') for i in df.precip]

df = df.drop(['events'], axis=1)

with open('weather_joined_utm.pkl', 'wb') as output:
    pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)


