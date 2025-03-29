import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def run_model(data):
    # =============================================================================
    # PART 1: Client-Provided Data from Flask JSON
    # =============================================================================
    client_annual_energy_conventional = float(data["client_annual_energy_conventional"])
    client_annual_water_conventional = float(data["client_annual_water_conventional"])
    client_avg_IT_load = float(data["client_avg_IT_load"])
    client_evaporation_rate = float(data["client_evaporation_rate"])
    client_energy_cost = float(data["client_energy_cost"])
    client_water_cost = float(data["client_water_cost"])
    client_ambient_temp = float(data["client_ambient_temp"])

    # Baseline
    conventional_cost = client_annual_energy_conventional * client_energy_cost + client_annual_water_conventional * client_water_cost
    conventional_energy_joules = client_annual_energy_conventional * 3.6e6

    T_list = list(range(12))
    IT_Load = {t: client_avg_IT_load for t in T_list}

    # Model parameters
    T_surface = client_ambient_temp
    alpha = 0.07
    T_dc_target = 27.0
    K_well = 2000.0
    g = 9.81
    rho = 1000.0
    Delta_t = 720.0
    eta = 0.7
    Q_max = 6.0
    D_min = 15.0
    D_max = 80.0
    V_initial = 1000.0
    leakage_rate = client_evaporation_rate
    cost_energy = client_energy_cost
    cost_water = client_water_cost
    B = T_dc_target - T_surface

    # Optimization model
    model = gp.Model("Optimal_Underground_Well_Yearly")
    model.Params.LogToConsole = 0  # Silence output

    Q = {}
    z = {}
    for t in T_list:
        Q[t] = model.addVar(lb=0.0, ub=Q_max, name=f"Q_{t}")
        z[t] = model.addVar(lb=0.0, ub=Q_max * D_max, name=f"z_{t}")

    D = model.addVar(lb=D_min, ub=D_max, name="Depth")
    model.update()

    for t in T_list:
        model.addConstr(K_well * (B * Q[t] + alpha * z[t]) >= IT_Load[t])

    for t in T_list:
        model.addConstr(z[t] >= D_min * Q[t])
        model.addConstr(z[t] >= Q_max * D + Q[t] * D_max - Q_max * D_max)
        model.addConstr(z[t] <= Q_max * D + Q[t] * D_min - Q_max * D_min)
        model.addConstr(z[t] <= D_max * Q[t])
    model.update()

    # Objective
    energy_expr = gp.quicksum((Q[t] * D * g * rho * Delta_t) / (1000 * eta) for t in T_list)
    water_pumped_expr = gp.quicksum(Q[t] * Delta_t for t in T_list)
    net_water_expr = V_initial + leakage_rate * water_pumped_expr
    energy_cost = energy_expr * cost_energy
    water_cost = net_water_expr * cost_water
    total_cost = energy_cost + water_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return {"error": "Optimisation failed", "status": model.status}

    # Final values
    optimal_D = D.X
    optimal_energy_new = energy_expr.getValue()
    optimal_net_water_new = net_water_expr.getValue()
    total_cost_new = model.ObjVal
    savings = conventional_cost - total_cost_new

    # Monthly values
    T_well_opt = T_surface - alpha * optimal_D
    flow_data = []
    cum_energy = 0.0
    cum_water = 0.0
    energy_arr = []
    water_arr = []

    for t in T_list:
        q = Q[t].X
        energy_t = (q * optimal_D * g * rho * Delta_t) / (1000 * eta)
        cum_energy += energy_t
        pumped = q * Delta_t
        cum_water += leakage_rate * pumped
        energy_arr.append(cum_energy)
        water_arr.append(V_initial + cum_water)

        T_dc_est = T_well_opt + IT_Load[t] / (K_well * q) if q > 1e-6 else None
        flow_data.append({
            "month": t,
            "flow_rate": q,
            "energy": energy_t,
            "estimated_T_dc": T_dc_est
        })

    # For plotting
    try:
        df_conv = pd.read_csv("conventional_datacentre_year.csv")
        df_conv['Cumulative_Water_Usage'] = df_conv['Makeup_Water_Usage_m3'].cumsum()
        df_conv['Cumulative_Energy'] = df_conv['Pump_Energy_kWh'].cumsum()
        time = df_conv['Time'].tolist()
        conv_energy = df_conv['Cumulative_Energy'].tolist()
        conv_water = df_conv['Cumulative_Water_Usage'].tolist()
    except:
        time, conv_energy, conv_water = [], [], []

    return {
        "optimal_energy_new": optimal_energy_new,
        "optimal_energy_joules": optimal_energy_new * 3.6e6,
        "optimal_net_water_new": optimal_net_water_new,
        "total_cost_new": total_cost_new,
        "client_annual_energy_conventional": client_annual_energy_conventional,
        "conventional_energy_joules": conventional_energy_joules,
        "client_annual_water_conventional": client_annual_water_conventional,
        "conventional_cost": conventional_cost,
        "savings": savings,
        "flow_data": flow_data,
        "cumulative_energy": energy_arr,
        "cumulative_water": water_arr,
        "conventional_energy": conv_energy,
        "conventional_water": conv_water,
        "months": time
    }
