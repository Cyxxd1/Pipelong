# model.py
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def run_model(inputs):
    # Extract user inputs
    surface_temp = float(inputs["surface_temp"])
    max_flow = float(inputs["max_flow"])
    initial_volume = float(inputs["initial_volume"])
    humidity = float(inputs["humidity"])
    wind_speed = float(inputs["wind_speed"])
    evap_override = inputs.get("evap_override")

    # Validate inputs
    if any(val < 0 for val in [surface_temp, max_flow, initial_volume, humidity, wind_speed]):
        raise ValueError("All input values must be non-negative.")
    if max_flow == 0:
        raise ValueError("Max pump flow rate must be greater than 0.")

    # Load datasets
    df_conv = pd.read_csv("data/conventional_datacentre.csv")
    df_well = pd.read_csv("data/underground_well_datacentre.csv")

    scaling_factor = 200
    total_energy_conv = df_conv['Pump_Energy_kWh'].sum()
    total_water_conv = (df_conv['Makeup_Water_Usage_m3'] * scaling_factor).sum()

    # Estimate or use override evaporation rate
    if evap_override:
        evap_rate_Lph = float(evap_override)
        leakage_rate = evap_rate_Lph / 1000
    else:
        df_conv['Evaporation_Rate_Lph'] = (
            df_conv['Evaporative_Loss_Percent'] / 100
            * df_conv['Pump_Flow_Rate_m3s']
            * 3600
            * 1000
        )
        evap_rate_Lph = df_conv['Evaporation_Rate_Lph'].mean()
        leakage_rate = evap_rate_Lph / 1000

    T_list = df_well['Time'].tolist()
    IT_Load = {int(row['Time']): row['IT_Load_kW'] for index, row in df_well.iterrows()}

    T_dc_target = 24.0
    alpha = 0.1
    K_well = 2000.0
    g = 9.81
    rho = 1000.0
    Delta_t = 1.0
    eta = 0.7

    # Bounds and costs
    Q_max = max_flow
    Q_min = 0.003
    D_min = 15.0
    D_max = 150.0
    cost_energy = 0.1
    cost_water = 1.0

    A = T_dc_target - surface_temp

    # Create model
    model = gp.Model("Underground_Well_Model")
    model.setParam("OutputFlag", 0)

    # Decision variables
    Q, z = {}, {}
    for t in T_list:
        Q[t] = model.addVar(lb=Q_min, ub=Q_max, name=f"Q_{t}")
        z[t] = model.addVar(lb=0.0, ub=Q_max * D_max, name=f"z_{t}")

    D = model.addVar(lb=D_min, ub=D_max, name="Depth")
    model.update()

    # Cooling constraints
    for t in T_list:
        model.addConstr(K_well * (A * Q[t] + alpha * z[t]) >= IT_Load[t], name=f"Cooling_{t}")
        model.addConstr(z[t] >= D_min * Q[t])
        model.addConstr(z[t] >= Q_max * D + Q[t] * D_max - Q_max * D_max)
        model.addConstr(z[t] <= Q_max * D + Q[t] * D_min - Q_max * D_min)
        model.addConstr(z[t] <= D_max * Q[t])

    # Objective components
    energy_expr = gp.quicksum((Q[t] * D * g * rho * Delta_t) / (1000 * eta) for t in T_list)
    energy_cost = energy_expr * cost_energy

    water_pumped_expr = gp.quicksum(Q[t] * Delta_t for t in T_list)
    net_water_expr = initial_volume + leakage_rate * water_pumped_expr
    water_cost = net_water_expr * cost_water

    total_cost = energy_cost + water_cost

    # Set objective and solve
    model.setObjective(total_cost, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise ValueError("Model did not find an optimal solution with the given parameters.")

    # Retrieve results
    optimal_D = D.X
    T_well_opt = surface_temp - alpha * optimal_D

    cumulative_energy_new = 0.0
    cumulative_IT_load = 0.0
    flow_rates = []
    last_T_dc = 0.0

    for t in T_list:
        q_val = Q[t].X
        it_load = IT_Load[t]
        energy_val = (q_val * optimal_D * g * rho * Delta_t) / (1000 * eta)
        cumulative_energy_new += energy_val
        cumulative_IT_load += it_load

        if q_val > 0:
            last_T_dc = T_well_opt + it_load / (K_well * q_val)

        flow_rates.append(q_val * 60000)

    avg_flow_rate = sum(flow_rates) / len(flow_rates) if flow_rates else 0.0
    reservoir_temp = T_well_opt

    water_saved = max((total_water_conv - net_water_expr.getValue()) * 1000, 0)

    cooling_output_kWh = cumulative_IT_load * Delta_t / 1000
    efficiency_percent = (cooling_output_kWh / cumulative_energy_new) * 100 if cumulative_energy_new > 0 else 0

    return {
        "gpu_temp": "70°C",
        "circulation_temp": f"{last_T_dc:.1f}°C",
        "reservoir_temp": f"{reservoir_temp:.1f}°C",
        "flow_rate": f"{avg_flow_rate:.1f} L/min",
        "efficiency": f"{efficiency_percent:.0f}%",
        "water_saved": f"{water_saved:.0f} L"
    }
