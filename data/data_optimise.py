import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: Client-Provided Conventional Baseline Data
# =============================================================================
# These values would come from the client's web form.
client_annual_energy_conventional = float(input("Enter your annual energy consumption for cooling (kWh/year): "))
client_annual_water_conventional = float(input("Enter your annual water consumption for cooling (m³/year): "))
client_avg_IT_load = float(input("Enter your average monthly IT load (kW): "))
client_evaporation_rate = float(input("Enter your evaporation rate (as a fraction, e.g., 0.05 for 5% loss per month): "))
client_energy_cost = float(input("Enter your energy cost ($/kWh): "))
client_water_cost = float(input("Enter your water cost ($/m³): "))
client_ambient_temp = float(input("Enter your average ambient temperature (°C): "))

# Compute conventional annual operating cost and convert energy to Joules.
conventional_cost = client_annual_energy_conventional * client_energy_cost + client_annual_water_conventional * client_water_cost
conventional_energy_joules = client_annual_energy_conventional * 3.6e6

print("\n=== Conventional System Baseline (Client Provided) ===")
print(f"Annual Energy Consumption: {client_annual_energy_conventional:.2f} kWh ({conventional_energy_joules:.2e} J)")
print(f"Annual Water Consumption: {client_annual_water_conventional:.2f} m³")
print(f"Annual Operating Cost: ${conventional_cost:.2f}")
print("-----------------------------------------------------\n")

# =============================================================================
# PART 2: Optimisation Model for Underground Well Cooling System (Yearly, 12 Months)
# =============================================================================
# For the new system, we assume a 12-month model.
# We use the client-provided average IT load for each month.
T_list = list(range(12))  # Months 0 to 11
IT_Load = {t: client_avg_IT_load for t in T_list}

# -------------------------------
# Model Parameters for Underground Well System
# -------------------------------
T_surface = client_ambient_temp   # Use client's ambient temperature as surface temperature
alpha = 0.07                      # Temperature drop per meter (°C/m)
T_dc_target = 27.0                # Target data centre temperature (°C)
K_well = 2000.0                   # Cooling coefficient (kW/(m³/s·°C))

g = 9.81                          # gravitational acceleration (m/s²)
rho = 1000.0                      # water density (kg/m³)
Delta_t = 720.0                   # hours per month (~30 days)
eta = 0.7

Q_max = 6.0                       # Maximum pump flow rate (m³/s)
D_min = 15.0                      # Minimum well depth (m)
D_max = 80.0                      # Maximum well depth (m) based on UQ Gatton findings

V_initial = 1000.0                # Initial water fill (m³)
leakage_rate = client_evaporation_rate  # Use client-provided evaporation rate

cost_energy = client_energy_cost  # $ per kWh
cost_water = client_water_cost    # $ per m³

# Define constant B = T_dc_target - T_surface
B = T_dc_target - T_surface       # e.g., if client_ambient_temp=25°C, then B = 2°C

# -------------------------------
# Create the Optimisation Model (Linearised Version)
# -------------------------------
model = gp.Model("Optimal_Underground_Well_Yearly")
# Decision variables for each month: monthly pump flow rate Q[t] and auxiliary variable z[t] = Q[t]*D
Q = {}
z = {}
for t in T_list:
    Q[t] = model.addVar(lb=0.0, ub=Q_max, name=f"Q_{t}")
    z[t] = model.addVar(lb=0.0, ub=Q_max * D_max, name=f"z_{t}")

# D: Well depth (m) is a design variable common to all months.
D = model.addVar(lb=D_min, ub=D_max, name="Depth")
model.update()

# -------------------------------
# Linearised Cooling Constraints for Each Month
# -------------------------------
# Original cooling constraint (per month):
#   T_dc = T_surface - alpha*D + IT_Load[t] / (K_well * Q[t]) <= T_dc_target.
# Rearranged: IT_Load[t] <= K_well * Q[t] * (T_dc_target - (T_surface - alpha*D))
# Let B = T_dc_target - T_surface, then:
#   IT_Load[t] <= K_well * Q[t] * (B + alpha*D)
# Multiply out:
#   K_well*(B*Q[t] + alpha*Q[t]*D) >= IT_Load[t]
# Replace Q[t]*D with auxiliary variable z[t]:
for t in T_list:
    model.addConstr(K_well * (B * Q[t] + alpha * z[t]) >= IT_Load[t],
                    name=f"Cooling_{t}")

# -------------------------------
# McCormick Envelope Constraints for z[t] = Q[t]*D
# -------------------------------
for t in T_list:
    model.addConstr(z[t] >= D_min * Q[t], name=f"McCormick1_{t}")
    model.addConstr(z[t] >= Q_max * D + Q[t] * D_max - Q_max * D_max, name=f"McCormick2_{t}")
    model.addConstr(z[t] <= Q_max * D + Q[t] * D_min - Q_max * D_min, name=f"McCormick3_{t}")
    model.addConstr(z[t] <= D_max * Q[t], name=f"McCormick4_{t}")
model.update()

# -------------------------------
# Define Monthly Energy and Water Usage Expressions
# -------------------------------
# Energy consumption per month (kWh):
energy_expr = gp.quicksum((Q[t] * D * g * rho * Delta_t) / (1000 * eta) for t in T_list)
energy_cost = energy_expr * cost_energy

# Total water pumped over the year (m³):
water_pumped_expr = gp.quicksum(Q[t] * Delta_t for t in T_list)
# Net water usage = initial fill + leakage losses (since water is recycled)
net_water_expr = V_initial + leakage_rate * water_pumped_expr
water_cost = net_water_expr * cost_water

# Total cost objective (minimise energy cost + water cost)
total_cost = energy_cost + water_cost
model.setObjective(total_cost, GRB.MINIMIZE)

# -------------------------------
# Solve the Model
# -------------------------------
model.optimize()

# -------------------------------
# Retrieve and Print Results for the Underground Well System (Yearly)
# -------------------------------
if model.status == GRB.OPTIMAL:
    optimal_D = D.X
    print("\nOptimal Underground Well Design (Yearly):")
    print(f"  Optimal Well Depth: {optimal_D:.2f} m")
    print("\nOptimal Monthly Pump Flow Rates and Estimated Data Centre Temperatures:")
    cumulative_energy_new = 0.0
    for t in T_list:
        q_val = Q[t].X
        energy_val = (q_val * optimal_D * g * rho * Delta_t) / (1000 * eta)
        cumulative_energy_new += energy_val
        
        # Compute well water temperature at optimal depth:
        T_well_opt = T_surface - alpha * optimal_D
        # Estimated monthly data centre temperature:
        T_dc_est = T_well_opt + IT_Load[t] / (K_well * q_val) if q_val > 1e-6 else 999
        print(f"Month {t}: Q = {q_val:.4f} m³/s, Energy = {energy_val:.2f} kWh, Estimated T_dc = {T_dc_est:.2f}°C")
    
    optimal_energy_new = energy_expr.getValue()
    optimal_net_water_new = net_water_expr.getValue()
    total_cost_new = model.ObjVal
    
    print("\nUnderground Well System (Yearly Totals):")
    print(f"  Total Pump Energy: {optimal_energy_new:.2f} kWh")
    print(f"  Total Pump Energy: {(optimal_energy_new * 3.6e6):.2e} J")
    print(f"  Net Water Usage (Initial fill + leakage): {optimal_net_water_new:.2f} m³")
    print(f"  Total Annual Cost: ${total_cost_new:.2f}")
    
    # =============================================================================
    # Yearly Comparison: Conventional vs. Underground Well
    # =============================================================================
    print("\n=== Yearly Comparison ===")
    print(f"Conventional System Energy: {client_annual_energy_conventional:.2f} kWh, {(conventional_energy_joules):.2e} J")
    print(f"Underground Well Energy: {optimal_energy_new:.2f} kWh, {(optimal_energy_new * 3.6e6):.2e} J")
    print(f"Conventional Water Usage: {client_annual_water_conventional:.2f} m³")
    print(f"Underground Well Water Usage: {optimal_net_water_new:.2f} m³")
    conventional_cost = client_annual_energy_conventional * cost_energy + client_annual_water_conventional * cost_water
    savings = conventional_cost - total_cost_new
    print(f"Annual Operating Cost (Conventional): ${conventional_cost:.2f}")
    print(f"Annual Operating Cost (Underground Well): ${total_cost_new:.2f}")
    print(f"Annual Cost Savings if switched: ${savings:.2f}")
    
else:
    print("No optimal solution found. Model status:", model.status)

# =============================================================================
# PART 3: Visualisation of Yearly Cumulative Comparisons
# =============================================================================
# Re-read conventional data to ensure df_conv is defined.
df_conv = pd.read_csv("conventional_datacentre_year.csv")
df_conv['Cumulative_Water_Usage'] = df_conv['Makeup_Water_Usage_m3'].cumsum()
df_conv['Cumulative_Energy'] = df_conv['Pump_Energy_kWh'].cumsum()

T_list_sorted = sorted(T_list)
cumulative_energy_new_arr = []
cumulative_water_new_arr = []
cum_energy_new = 0.0
cum_leakage_new = 0.0

for t in T_list_sorted:
    q_val = Q[t].X
    energy_val = (q_val * optimal_D * g * rho * Delta_t) / (1000 * eta)
    cum_energy_new += energy_val
    cumulative_energy_new_arr.append(cum_energy_new)
    
    pumped_water = q_val * Delta_t
    cum_leakage_new += leakage_rate * pumped_water
    cumulative_water_new_arr.append(V_initial + cum_leakage_new)

plt.figure(figsize=(10, 6))
plt.plot(df_conv['Time'], df_conv['Cumulative_Water_Usage'], marker='o', label="Conventional Water Usage")
plt.plot(T_list_sorted, cumulative_water_new_arr, marker='s', linestyle='--', label="Underground Well Water Usage")
plt.xlabel("Month (0-11)")
plt.ylabel("Cumulative Water Usage (m³)")
plt.title("Yearly Cumulative Water Usage Comparison")
plt.legend()
plt.grid(True)
plt.xticks(df_conv['Time'])
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_conv['Time'], df_conv['Cumulative_Energy'], marker='o', label="Conventional Energy (kWh)")
plt.plot(T_list_sorted, cumulative_energy_new_arr, marker='s', linestyle='--', label="Underground Well Energy (kWh)")
plt.xlabel("Month (0-11)")
plt.ylabel("Cumulative Energy Consumption (kWh)")
plt.title("Yearly Cumulative Energy Consumption Comparison")
plt.legend()
plt.grid(True)
plt.xticks(df_conv['Time'])
plt.show()
