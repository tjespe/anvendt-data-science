#  %%[markdown]
# # Visulatization of data


# %%
from preprocessing import read_consumption_data
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = read_consumption_data()

# Graph of Temperature development
plt.xlabel('Time')
plt.ylabel('Temprature')
plt.title('Temperature for each day')
plt.plot(df["time"], df["temperature"], 'r')


# Graph of Energy Consumption development
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.title('Energy Consumption per day')
plt.plot(df["time"], df["consumption"], 'r')


# %%
