import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load the Datasets
# -------------------------------
# Make sure you have saved the CSV texts as "conventional_datacentre.csv" and "underground_well_datacentre.csv".
df_conv = pd.read_csv("data/conventional_datacentre.csv")
df_well = pd.read_csv("data/underground_well_datacentre.csv")

# -------------------------------
# Assumptions and Scaling Factors
# -------------------------------
# Conventional system:
# Our synthetic conventional dataset has very small makeup water usage (e.g. ~0.0036 m³/h).
# In a real, large data centre, the water usage might be significantly higher.
# We introduce a scaling factor to bring these values into a realistic range.
scaling_factor = 2000   # e.g. multiply synthetic water usage by 2000 to get realistic values (m³/h)

# Underground well system:
# Assume the initial water fill required in the well is V_initial (m³).
V_initial = 100.0       # Assume we need to initially fill the well with 100 m³ of water.

# -------------------------------
# Compute Cumulative Water Usage Over 24 Hours
# -------------------------------
# For the conventional system, cumulative water usage is the sum of the (scaled) makeup water usage.
df_conv['Scaled_Makeup_Water'] = df_conv['Makeup_Water_Usage_m3'] * scaling_factor
df_conv['Cumulative_Water_Usage'] = df_conv['Scaled_Makeup_Water'].cumsum()

# For the underground well system, since water is recycled, the net water usage after the initial fill is zero.
# Therefore, cumulative water usage remains constant at V_initial.
# We create a column with this constant value.
df_well['Cumulative_Water_Usage'] = V_initial

# Determine the break-even time (hour) when conventional cumulative water usage exceeds V_initial.
break_even_hour = None
for idx, row in df_conv.iterrows():
    if row['Cumulative_Water_Usage'] >= V_initial:
        break_even_hour = row['Time']
        break

# -------------------------------
# Compute Cumulative Energy Consumption (Optional)
# -------------------------------
# For simplicity, we sum the pump energy consumption directly from each dataset.
# (In a real scenario, you might also need a scaling factor for energy or use actual measurements.)
df_conv['Cumulative_Energy'] = df_conv['Pump_Energy_kWh'].cumsum()
df_well['Cumulative_Energy'] = df_well['Pump_Energy_kWh'].cumsum()

# -------------------------------
# Print Comparison Summary
# -------------------------------
print("Conventional Water-Cooled Data Centre (Scaled) over 24 Hours:")
print(df_conv[['Time', 'Scaled_Makeup_Water', 'Cumulative_Water_Usage']])
print("\nUnderground Well Cooling Solution:")
print(df_well[['Time', 'Cumulative_Water_Usage']])

print("\nBreak-even Hour (when conventional system's cumulative water usage exceeds the initial fill):")
if break_even_hour is not None:
    print(f"Break-even time: Hour {break_even_hour}")
else:
    print("The conventional system did not exceed the initial water fill in 24 hours.")

print("\nTotal Energy Consumption over 24 Hours:")
total_energy_conv = df_conv['Pump_Energy_kWh'].sum()
total_energy_well = df_well['Pump_Energy_kWh'].sum()
print(f"Conventional System: {total_energy_conv:.2f} kWh")
print(f"Underground Well System: {total_energy_well:.2f} kWh")

# -------------------------------
# Visualisation: Cumulative Water Usage
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_conv['Time'], df_conv['Cumulative_Water_Usage'], marker='o', label='Conventional (Scaled)')
plt.plot(df_well['Time'], df_well['Cumulative_Water_Usage'], marker='s', linestyle='--', label='Underground Well')
plt.xlabel("Time (Hour)")
plt.ylabel("Cumulative Water Usage (m³)")
plt.title("Cumulative Water Usage Comparison Over 24 Hours")
plt.legend()
plt.grid(True)
plt.xticks(df_conv['Time'])
plt.show()

# -------------------------------
# Visualisation: Cumulative Energy Consumption
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_conv['Time'], df_conv['Cumulative_Energy'], marker='o', label='Conventional Energy (kWh)')
plt.plot(df_well['Time'], df_well['Cumulative_Energy'], marker='s', linestyle='--', label='Underground Well Energy (kWh)')
plt.xlabel("Time (Hour)")
plt.ylabel("Cumulative Energy Consumption (kWh)")
plt.title("Cumulative Energy Consumption Over 24 Hours")
plt.legend()
plt.grid(True)
plt.xticks(df_conv['Time'])
plt.show()
