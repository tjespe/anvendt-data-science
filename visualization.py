#  %%[markdown]
# # Visulatization of data


# %%
from preprocessing import read_consumption_data
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = read_consumption_data()

# %%
# Graph of Temperature development

daily_avg_temp = (
    df.groupby([df["time"].dt.date, "location"])["temperature"].mean().unstack()
)
# Loop through each city and plot its daily average temperature
for city in daily_avg_temp.columns:
    plt.plot(daily_avg_temp.index, daily_avg_temp[city], label=city)

# Customize the plot
plt.title("Daily Average Temperature")
plt.xlabel("Date")
plt.ylabel("Average Temperature")
plt.legend()
plt.grid(True)

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Histogram of all temperature values in dataset
plt.hist(df["temperature"], bins=100)
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title("Histogram of Temperature")
plt.show()

# %%
# Graph of Energy Consumption development
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.title("Energy Consumption per day")
plt.plot(df["time"], df["consumption"], "r")


# %%
