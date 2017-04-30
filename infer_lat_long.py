import pandas as pd
import pickle
import numpy as np

df = pd.read_csv("./NYPD_Motor_Vehicle_Collisions.csv")
df = df[['BOROUGH', 'ZIP CODE', 'LATITUDE', 'LONGITUDE', 'LOCATION', 'ON STREET NAME', 
         'CROSS STREET NAME', 'OFF STREET NAME','UNIQUE KEY']]
df['Infered_Lat'] = np.NaN
df['Infered_Long'] = np.NaN
df['Lat/Long Infered'] = False

nans = df[df.LATITUDE.isnull()].copy()

for i in range(len(nans)):
    if i % 100 ==0:
        print(i)

    (on, cross, off) = nans.iloc[i][['ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME']].values

    if not(any([pd.isnull(on) , pd.isnull(cross)])):
        (infered_lat, infered_long) = df[(df['ON STREET NAME']==on) & (df['CROSS STREET NAME']==cross)][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            nans.iloc[i][['Infered_Lat', 'Infered_Long','Lat/Long Infered']]=(infered_lat, infered_long, 'On_Cross')
            continue

    if not(pd.isnull(on)):
        (infered_lat, infered_long) = df[df['ON STREET NAME']==on][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            nans.iloc[i][['Infered_Lat', 'Infered_Long','Lat/Long Infered']]=(infered_lat, infered_long, 'On')
            continue

    if not(pd.isnull(off)):
        (infered_lat, infered_long) = df[df['ON STREET NAME']==off][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            nans.iloc[i][['Infered_Lat', 'Infered_Long','Lat/Long Infered']]=(infered_lat, infered_long, 'Off')
            continue

    if not(pd.isnull(cross)):
        (infered_lat, infered_long) = df[df['ON STREET NAME']==cross][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            nans.iloc[i][['Infered_Lat', 'Infered_Long','Lat/Long Infered']]=(infered_lat, infered_long, 'Cross')
            continue

pickle.dump(open('./infered.pkl','wb'),nans)
                
