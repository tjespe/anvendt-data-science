#  %%[markdown]
# # Visulatization of data


# %%
from preprocessing import read_consumption_data
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = read_consumption_data()

# %%
# Graph of daily average temperature

daily_avg_temp = (
    df.groupby([df["Time"].dt.date, "Location"])["Temperature"].mean().unstack()
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
plt.hist(df["Temperature"], bins=100)
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title("Histogram of Temperature")
plt.show()


# %%
# Graph of energy consumption development
daily_avg_energy_consumption = (
    df.groupby([df["Time"].dt.date, "Consumption"])["Consumption"].mean().unstack()
)
# Loop through each city and plot its daily average energy consumption
for city in daily_avg_energy_consumption.columns:
    plt.plot(
        daily_avg_energy_consumption.index,
        daily_avg_energy_consumption[city],
        label=city,
    )

# Customize the plot
plt.title("Daily Average Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Average Energy Consumption")
plt.legend()
plt.grid(True)

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Histogram of all energy consumption values in dataset
plt.hist(df["Consumption"], bins=100)
plt.xlabel("Consumption")
plt.ylabel("Frequency")
plt.title("Histogram of Energy consumption")
plt.show()

# %%
