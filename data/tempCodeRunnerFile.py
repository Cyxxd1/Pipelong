import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: Client-Provided Conventional Baseline Data (Set Values)
# =============================================================================
# Set fixed values
client_annual_energy_conventional = 500000.0    # kWh/year
client_annual_water_conventional = 10000.0       # m³/year
client_avg_IT_load = 3000.0                       # kW (average monthly IT load)
client_evaporation_rate = 0.05                   # 5% evaporation rate per month
client_energy_cost = 0.10                        # $ per kWh
client_water_cost = 2.00                         # $ per m³
client_ambient_temp = 25.0                       # °C

# Compute conventional annual operating cost and convert energy to Joules.
conventional_cost = (client_annual_energy_conventional * client_energy_cost +
                     client_annual_water_conventional * client_water_cost)
conventional_energy_joules = client_annual_energy_conventional * 3.6e6

print("\n=== Conventional System Baseline (Set Values) ===")
print(f"Annual Energy Consumption: {client_annual_energy_conventional:.2f} kWh "
      f"({conventional_energy_joules:.2e} J)")
print(f"Annual Water Consumption: {client_annual_water_conventional:.2f} m³")
print(f"Annual Operating Cost: ${conventional_cost:.2f}")
print("-----------------------------------------------------\n")

# =============================================================================
# PART 2: Optimisation Model for Underground Well Cooling System (Yearly, 12 Months)
# =============================================================================
T_list = list(range(12))  # Months 0 to 11
IT_Load = {t: client_avg_IT_load for t in T_list}

# Model parameters for the underground well system
T_surface = client_ambient_temp   # Using the set ambient temperature as surface temperature
alpha = 0.07                      # Temperature drop per meter (°C/m)
T_dc_target = 27.0                # Target data centre temperature (°C)
K_well = 2000.0                   # Cooling coefficient (kW/(m³/s·°C))

g = 9.81                          # gravitational acceleration (m/s²)
rho = 1000.0                      # water density (kg/m³)
Delta_t = 720.0                   # hours per month (~30 days)
eta = 0.7

Q_max = 6.0                       # Maximum pump flow rate (m³/s)
D_min = 15.0                      # Minimum well depth (m)
D_max = 80.0                      # Maximum well depth (m)

V_initial = 6000.0                # Initial water fill (m³)
leakage_rate = client_evaporation_rate  # Evaporation rate

cost_energy = client_energy_cost  # Energy cost per kWh
cost_water = client_water_cost    # Water cost per m³

# Define constant B = T_dc_target - T_surface
B = T_dc_target - T_surface

# Create the Optimisation Model (Linearised Version)
model = gp.Model("Optimal_Underground_Well_Yearly")
# Decision variables: monthly pump flow rate Q[t] and auxiliary variable z[t] = Q[t] * D
Q = {}
z = {}
for t in T_list:
    Q[t] = model.addVar(lb=0.0, ub=Q_max, name=f"Q_{t}")
    z[t] = model.addVar(lb=0.0, ub=Q_max * D_max, name=f"z_{t}")

# D: Well depth (m) is a design variable common to all months.
D = model.addVar(lb=D_min, ub=D_max, name="Depth")
model.update()

# Linearised Cooling Constraints for each month:
# IT_Load[t] <= K_well * Q[t] * (B + alpha*D) is reformulated as:
# K_well * (B * Q[t] + alpha * (Q[t]*D)) >= IT_Load[t]
for t in T_list:
    model.addConstr(K_well * (B * Q[t] + alpha * z[t]) >= IT_Load[t],
                    name=f"Cooling_{t}")

# McCormick Envelope Constraints for z[t] = Q[t] * D
for t in T_list:
    model.addConstr(z[t] >= D_min * Q[t], name=f"McCormick1_{t}")
    model.addConstr(z[t] >= Q_max * D + Q[t] * D_max - Q_max * D_max, name=f"McCormick2_{t}")
    model.addConstr(z[t] <= Q_max * D + Q[t] * D_min - Q_max * D_min, name=f"McCormick3_{t}")
    model.addConstr(z[t] <= D_max * Q[t], name=f"McCormick4_{t}")
model.update()

# Define monthly energy and water usage expressions
energy_expr = gp.quicksum((Q[t] * D * g * rho * Delta_t) / (1000 * eta) for t in T_list)
energy_cost_expr = energy_expr * cost_energy

water_pumped_expr = gp.quicksum(Q[t] * Delta_t for t in T_list)
# Net water usage = initial fill + leakage losses (water is recycled)
net_water_expr = V_initial + leakage_rate * water_pumped_expr
water_cost_expr = net_water_expr * cost_water

# Total cost objective (minimise energy cost + water cost)
total_cost = energy_cost_expr + water_cost_expr
model.setObjective(total_cost, GRB.MINIMIZE)

# Solve the Model
model.optimize()

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
        # Compute well water temperature at optimal depth
        T_well_opt = T_surface - alpha * optimal_D
        # Estimated data centre temperature for month t
        T_dc_est = T_well_opt + IT_Load[t] / (K_well * q_val) if q_val > 1e-6 else 999
        print(f"Month {t}: Q = {q_val:.4f} m³/s, Energy = {energy_val:.2f} kWh, "
              f"Estimated T_dc = {T_dc_est:.2f}°C")
    
    optimal_energy_new = energy_expr.getValue()
    optimal_net_water_new = net_water_expr.getValue()
    total_cost_new = model.ObjVal
    
    print("\nUnderground Well System (Yearly Totals):")
    print(f"  Total Pump Energy: {optimal_energy_new:.2f} kWh "
          f"({optimal_energy_new * 3.6e6:.2e} J)")
    print(f"  Net Water Usage (Initial fill + leakage): {optimal_net_water_new:.2f} m³")
    print(f"  Total Annual Cost: ${total_cost_new:.2f}")
    
    print("\n=== Yearly Comparison ===")
    print(f"Conventional System Energy: {client_annual_energy_conventional:.2f} kWh, "
          f"{conventional_energy_joules:.2e} J")
    print(f"Underground Well Energy: {optimal_energy_new:.2f} kWh, "
          f"{(optimal_energy_new * 3.6e6):.2e} J")
    print(f"Conventional Water Usage: {client_annual_water_conventional:.2f} m³")
    print(f"Underground Well Water Usage: {optimal_net_water_new:.2f} m³")
    conventional_cost = (client_annual_energy_conventional * client_energy_cost +
                         client_annual_water_conventional * client_water_cost)
    savings = conventional_cost - total_cost_new
    print(f"Annual Operating Cost (Conventional): ${conventional_cost:.2f}")
    print(f"Annual Operating Cost (Underground Well): ${total_cost_new:.2f}")
    print(f"Annual Cost Savings if switched: ${savings:.2f}")
    
else:
    print("No optimal solution found. Model status:", model.status)

# =============================================================================
# PART 3: Monthly Cumulative Comparison for Operating Cost and Water Usage
# =============================================================================
# For the conventional system, assume values are evenly distributed over 12 months.
months = list(range(12))
conv_monthly_energy = client_annual_energy_conventional / 12
conv_monthly_water = client_annual_water_conventional / 12
conv_monthly_energy_cost = conv_monthly_energy * client_energy_cost
conv_monthly_water_cost = conv_monthly_water * client_water_cost
conv_monthly_total_cost = conv_monthly_energy_cost + conv_monthly_water_cost

conv_cum_cost = []
conv_cum_water = []
cum_cost = 0.0
cum_water = 0.0
for m in months:
    cum_cost += conv_monthly_total_cost
    conv_cum_cost.append(cum_cost)
    cum_water += conv_monthly_water
    conv_cum_water.append(cum_water)

# For the underground well system, compute monthly cumulative values using the optimisation results.
new_cum_cost = []
new_cum_water = []
cum_cost_new = 0.0
cum_water_new = 0.0

for t in months:
    q_val = Q[t].X
    energy_val = (q_val * optimal_D * g * rho * Delta_t) / (1000 * eta)
    energy_cost_new = energy_val * client_energy_cost

    # Water usage in month t (including leakage)
    water_pumped = q_val * Delta_t
    if t == 0:
        water_usage_new = V_initial + leakage_rate * water_pumped
        water_cost_new = (V_initial * client_water_cost) + (leakage_rate * water_pumped * client_water_cost)
    else:
        water_usage_new = leakage_rate * water_pumped
        water_cost_new = leakage_rate * water_pumped * client_water_cost

    total_cost_new_month = energy_cost_new + water_cost_new
    
    cum_cost_new += total_cost_new_month
    new_cum_cost.append(cum_cost_new)
    
    cum_water_new += water_usage_new
    new_cum_water.append(cum_water_new)

# =============================================================================
# Plot 1: Cumulative Operating Cost Comparison
# =============================================================================
plt.figure(figsize=(10, 6))
plt.plot(months, conv_cum_cost, marker='o', label="Conventional Operating Cost")
plt.plot(months, new_cum_cost, marker='s', linestyle='--', label="Underground Well Operating Cost")
plt.xlabel("Month (0-11)")
plt.ylabel("Cumulative Operating Cost ($)")
plt.title("Cumulative Operating Cost Comparison")
plt.legend()
plt.grid(True)
plt.xticks(months)
plt.show()

# =============================================================================
# Plot 2: Cumulative Water Usage Comparison
# =============================================================================
plt.figure(figsize=(10, 6))
plt.plot(months, conv_cum_water, marker='o', label="Conventional Water Usage")
plt.plot(months, new_cum_water, marker='s', linestyle='--', label="Underground Well Water Usage")
plt.xlabel("Month (0-11)")
plt.ylabel("Cumulative Water Usage (m³)")
plt.title("Cumulative Water Usage Comparison")
plt.legend()
plt.grid(True)
plt.xticks(months)
plt.show()
