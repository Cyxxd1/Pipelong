import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ====================================================
# Part 1: Load Conventional System Data and Compute Totals
# ====================================================
df_conv = pd.read_csv("data/conventional_datacentre.csv")
scaling_factor = 2000   # Scale factor to bring synthetic water usage to realistic values

# Cumulative conventional energy and water usage (over 24 hours)
total_energy_conv = df_conv['Pump_Energy_kWh'].sum()
total_water_conv = (df_conv['Makeup_Water_Usage_m3'] * scaling_factor).sum()

print("Conventional Water-Cooled Data Centre (24 Hours):")
print(f"  Total Pump Energy Consumption: {total_energy_conv:.2f} kWh")
print(f"  Total Water Usage (Makeup water): {total_water_conv:.2f} m³")
print("-----------------------------------------------------\n")

# ====================================================
# Part 2: Set Up Optimisation Model for the Underground Well Solution
# ====================================================
# Load the underground well dataset (which provides hourly IT load data)
df_well = pd.read_csv("data/underground_well_datacentre.csv")
T_list = df_well['Time'].tolist()  # Assume times 0 to 23
IT_Load = {int(row['Time']): row['IT_Load_kW'] for index, row in df_well.iterrows()}

# -------------------------------
# Define Model Parameters
# -------------------------------
T_surface = 25.0         # Surface temperature (°C)
alpha = 0.1              # Temperature drop per meter (°C/m)
# Well water temperature: T_well = T_surface - alpha*D

T_dc_target = 24.0       # Target (maximum) data centre temperature (°C)
K_well = 2000.0          # Cooling coefficient (kW per (m³/s·°C))

# Physical parameters for energy calculation:
g = 9.81                 # gravitational acceleration (m/s²)
rho = 1000.0             # density of water (kg/m³)
Delta_t = 1.0            # time period duration (hour)
eta = 0.7                # pump efficiency

Q_max = 0.05             # Maximum pump flow rate (m³/s)
# For each hour, Q[t] will be in [0, Q_max].

# Design variable for well depth:
D_min = 15.0             # Minimum well depth (m)
D_max = 150.0            # Maximum well depth (m)

# Water recycling parameters:
V_initial = 100.0        # Initial water fill (m³)
leakage_rate = 0.01      # Fraction of pumped water lost per hour

# Cost factors:
cost_energy = 0.1        # $ per kWh
cost_water = 1.0         # $ per m³

# Define constant A = T_dc_target - T_surface
A = T_dc_target - T_surface  # e.g., 24 - 25 = -1.
# Note: With a deep well, T_well = T_surface - alpha*D will be much lower than T_surface.

# -------------------------------
# Create the Optimisation Model (Linearised Version)
# -------------------------------
model = gp.Model("Optimal_Underground_Well_FullModel")

# Enable nonconvex mode if needed (we are linearising the bilinear term so it should be linear)
# model.Params.NonConvex = 2

# Decision Variables:
# For each time period t, define Q[t] (pump flow rate)
Q = {}
# Also define auxiliary variable z[t] = Q[t] * D
z = {}
for t in T_list:
    Q[t] = model.addVar(lb=0.0, ub=Q_max, name=f"Q_{t}")
    z[t] = model.addVar(lb=0.0, ub=Q_max * D_max, name=f"z_{t}")

# D: Well depth (m)
D = model.addVar(lb=D_min, ub=D_max, name="Depth")

model.update()

# -------------------------------
# Linearised Cooling Constraints:
# Original cooling constraint for each hour t:
#   T_dc = T_surface - alpha*D + IT_Load[t]/(K_well*Q[t]) <= T_dc_target.
# Rearranging: IT_Load[t] <= K_well * Q[t] * (T_dc_target - (T_surface - alpha*D))
# Let A = T_dc_target - T_surface, then:
#   IT_Load[t] <= K_well * Q[t] * (A + alpha*D)
# Multiply out and rewrite as:
#   K_well * (A * Q[t] + alpha * (Q[t]*D)) >= IT_Load[t]
# Replace Q[t]*D by z[t]:
for t in T_list:
    model.addConstr(K_well * (A * Q[t] + alpha * z[t]) >= IT_Load[t], name=f"Cooling_{t}")

# -------------------------------
# McCormick Envelopes for z[t] = Q[t] * D
# -------------------------------
# For each t, with Q[t] in [0, Q_max] and D in [D_min, D_max]
for t in T_list:
    model.addConstr(z[t] >= D_min * Q[t], name=f"McCormick1_{t}")
    model.addConstr(z[t] >= Q_max * D + Q[t] * D_max - Q_max * D_max, name=f"McCormick2_{t}")
    model.addConstr(z[t] <= Q_max * D + Q[t] * D_min - Q_max * D_min, name=f"McCormick3_{t}")
    model.addConstr(z[t] <= D_max * Q[t], name=f"McCormick4_{t}")

model.update()

# -------------------------------
# Define Energy and Water Usage Expressions for the Underground Well System
# -------------------------------
# Energy consumption for each hour: E[t] = (Q[t] * D * g * rho * Delta_t)/(1000*eta)
energy_expr = gp.quicksum((Q[t] * D * g * rho * Delta_t) / (1000 * eta) for t in T_list)
energy_cost = energy_expr * cost_energy

# Total water pumped over the day:
water_pumped_expr = gp.quicksum(Q[t] * Delta_t for t in T_list)
# Net water usage: initial fill + leakage losses
net_water_expr = V_initial + leakage_rate * water_pumped_expr
water_cost = net_water_expr * cost_water

# Total cost objective: energy cost + water cost
total_cost = energy_cost + water_cost
model.setObjective(total_cost, GRB.MINIMIZE)

# -------------------------------
# Solve the Model
# -------------------------------
model.optimize()

# -------------------------------
# Retrieve and Print the Results for the Underground Well Model
# -------------------------------
if model.status == GRB.OPTIMAL:
    optimal_D = D.X
    print("\nOptimal Underground Well Design (New System):")
    print(f"  Optimal Well Depth: {optimal_D:.2f} m")
    print("\nOptimal Pump Flow Rates per Hour:")
    cumulative_energy_new = 0.0
    cumulative_water_new = 0.0
    for t in T_list:
        q_val = Q[t].X
        energy_val = (q_val * optimal_D * g * rho * Delta_t) / (1000 * eta)
        cumulative_energy_new += energy_val
        pumped_water = q_val * Delta_t
        cumulative_water_new += pumped_water
        # Calculate well water temperature from optimal depth:
        T_well_opt = T_surface - alpha * optimal_D
        # Estimated data centre temperature:
        T_dc_est = T_well_opt + IT_Load[t] / (K_well * q_val)
        print(f"Time {t}: Q = {q_val:.4f} m³/s, Energy = {energy_val:.4f} kWh, Pumped Water = {pumped_water:.4f} m³, Estimated T_dc = {T_dc_est:.2f}°C")
    
    optimal_energy_new = energy_expr.getValue()
    optimal_net_water_new = net_water_expr.getValue()
    total_cost_new = model.ObjVal
    
    print("\nUnderground Well System (24 Hours):")
    print(f"  Total Pump Energy Consumption: {optimal_energy_new:.2f} kWh")
    print(f"  Net Water Usage (Initial fill + leakage): {optimal_net_water_new:.2f} m³")
    print(f"  Total Cost: ${total_cost_new:.2f}")
    
    # ====================================================
    # Comparison: Conventional vs Underground Well
    # ====================================================
    print("\n=== 24-Hour Comparison ===")
    print(f"Conventional System Energy Consumption: {total_energy_conv:.2f} kWh")
    print(f"Underground Well Energy Consumption: {optimal_energy_new:.2f} kWh")
    print(f"Conventional System Water Usage: {total_water_conv:.2f} m³")
    print(f"Underground Well Net Water Usage: {optimal_net_water_new:.2f} m³")
    
else:
    print("No optimal solution found. Model status:", model.status)

# ====================================================
# Part 3: Visualisation of Cumulative Comparisons Over 24 Hours
# ====================================================
# For the conventional system, compute cumulative water usage and energy.
df_conv['Cumulative_Water_Usage'] = (df_conv['Makeup_Water_Usage_m3'] * scaling_factor).cumsum()
df_conv['Cumulative_Energy'] = df_conv['Pump_Energy_kWh'].cumsum()

# For the new (underground well) system, build cumulative arrays from the optimal solution.
cumulative_energy_new_arr = []
cumulative_water_new_arr = []
cum_energy_new = 0.0
cum_leakage_new = 0.0
for t in T_list:
    q_val = Q[t].X
    energy_val = (q_val * optimal_D * g * rho * Delta_t) / (1000 * eta)
    cum_energy_new += energy_val
    cumulative_energy_new_arr.append(cum_energy_new)
    
    pumped_water = q_val * Delta_t
    cum_leakage_new += leakage_rate * pumped_water
    cumulative_water_new_arr.append(V_initial + cum_leakage_new)

plt.figure(figsize=(10, 6))
plt.plot(df_conv['Time'], df_conv['Cumulative_Water_Usage'], marker='o', label="Conventional Water Usage (Scaled)")
plt.plot(df_well['Time'], cumulative_water_new_arr, marker='s', linestyle='--', label="Underground Well Net Water Usage")
plt.xlabel("Time (Hour)")
plt.ylabel("Cumulative Water Usage (m³)")
plt.title("Cumulative Water Usage Comparison Over 24 Hours")
plt.legend()
plt.grid(True)
plt.xticks(df_conv['Time'])
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_conv['Time'], df_conv['Cumulative_Energy'], marker='o', label="Conventional Energy (kWh)")
plt.plot(df_well['Time'], cumulative_energy_new_arr, marker='s', linestyle='--', label="Underground Well Energy (kWh)")
plt.xlabel("Time (Hour)")
plt.ylabel("Cumulative Energy Consumption (kWh)")
plt.title("Cumulative Energy Consumption Comparison Over 24 Hours")
plt.legend()
plt.grid(True)
plt.xticks(df_conv['Time'])
plt.show()
