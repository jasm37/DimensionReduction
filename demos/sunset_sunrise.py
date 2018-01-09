import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diff_map import DiffusionMap

dfs = pd.read_excel('data_suntimes.xlsx', sheetname=None)
cities = dfs['Tabelle1']['Cities'].tolist()
sunrise = dfs['Tabelle1']['Sunrise'].as_matrix()
sunset = dfs['Tabelle1']['Sunset'].as_matrix()
latitude = dfs['Tabelle1']['Latitude'].as_matrix()
longitude = dfs['Tabelle1']['Longitude'].as_matrix()


joint_data = np.vstack((sunrise,sunset)).T

ndim = 2
diff_map = DiffusionMap()
diff_map.set_params(joint_data)
w, x = diff_map.dim_reduction(ndim)

'''
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(211)
ax1.set_title("Ambient Space")

i=0
for xy in joint_data:
    ax1.annotate(cities[i],xy=xy, textcoords='data')
    i+=1

plt.scatter(joint_data[:, 0], joint_data[:, 1], c=latitude, cmap=plt.cm.jet)
plt.colorbar()

ax2 = fig.add_subplot(212)
ax2.set_title("Diffusion Maps")
'''
fig = plt.figure(figsize=(10,10))
ax2 = fig.add_subplot(111)
ax2.set_title("Diffusion Maps")

i=0
x = x * [-1,1] # Give proper orientation, proportional to map locations
for xy in x:
    ax2.annotate(cities[i],xy=xy, textcoords='data')
    i+=1
plt.scatter(x[:, 0], x[:, 1],c=latitude , cmap=plt.cm.jet)
plt.colorbar().set_label('Latitude')
plt.show()
