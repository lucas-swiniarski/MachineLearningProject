import pandas as pd
import pickle
import numpy as np

# df = pd.read_csv("~/Google Drive/Data_Science/NYU/Machine Learning/ML Project (Collisions)/NYPD_Motor_Vehicle_Collisions.csv")
df = pd.read_csv("./NYPD_Motor_Vehicle_Collisions.csv")
df = df[['BOROUGH', 'ZIP CODE', 'LATITUDE', 'LONGITUDE', 'LOCATION', 'ON STREET NAME',
         'CROSS STREET NAME', 'OFF STREET NAME','UNIQUE KEY']]
df['Infered_Lat'] = np.NaN
df['Infered_Long'] = np.NaN
df['Lat/Long Infered'] = False

nans = df[df.LATITUDE.isnull()]
output = np.array(['UNIQUE KEY', 'Infered_Lat','Infered_Long','Lat/Long Infered'])
for i in range(len(nans)):
# for i in range(500):
    if i % 100 ==0:
        print(i)

    (on, cross, off, key) = nans.iloc[i][['ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME','UNIQUE KEY']].values
    on_isnan = pd.isnull(on)
    cross_isnan = pd.isnull(cross)
    off_isnan = pd.isnull(off)

    if all([on_isnan, cross_isnan, off_isnan]):
        continue

    if not(any([on_isnan , cross_isnan])):
        (infered_lat, infered_long) = df[(df['ON STREET NAME']==on) & (df['CROSS STREET NAME']==cross)][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            output = np.vstack([output,[key, infered_lat, infered_long, 'On_Cross']])
            continue

    if not(on_isnan):
        (infered_lat, infered_long) = df[df['ON STREET NAME']==on][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            output = np.vstack([output,[key, infered_lat, infered_long, 'On']])
            continue

    if not(off_isnan):
        (infered_lat, infered_long) = df[df['ON STREET NAME']==off][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            output = np.vstack([output,[key, infered_lat, infered_long, 'Off']])
            continue

    if not(cross_isnan):
        (infered_lat, infered_long) = df[df['ON STREET NAME']==cross][['LATITUDE','LONGITUDE']].mean()
        if not(pd.isnull(infered_lat)):
            output = np.vstack([output,[key, infered_lat, infered_long, 'Cross']])
            continue

pickle.dump(pd.DataFrame(output[1:,1:],index=output[1:,0], columns = output[0,1:]),open('./infered.pkl','wb'),)
