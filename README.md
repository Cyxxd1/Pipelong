# Pipelong
# Geothermal-Assisted AI Data Centre Cooling

## Project Summary
AI data centres consume large volumes of water for cooling, primarily through evaporative methods that result in significant water loss. As AI infrastructure expands, this issue is set to grow.

Our project introduces a **closed-loop geothermal cooling system** that significantly reduces water consumption by:
- Circulating a small amount of water (15L) to cool GPUs
- Directing the heated water to a **300L underground reservoir**, passively cooled by the earth
- Using natural geothermal properties (similar to underground caves) to cool without evaporation

The solution enhances **water conservation**, boosts **energy efficiency**, and offers a **scalable, sustainable** approach for future AI data centres.

---

## 🖥️ App Overview
This repository includes a working simulation and dashboard for the system:
- Displays **real-time simulated values** for:
  - GPU temperature
  - Circulation water temperature
  - Reservoir temperature
  - Flow rate
  - Energy efficiency
  - Water saved
  - Uses an optimisation model (Gurobi) to compute ideal pump flow rates and well depth
  - Compares results with traditional evaporative cooling data

---

## 🛠️ Tech Stack
- **Python** (Flask backend)
- **Gurobi** (optimisation model)
- **HTML/CSS/JS** (frontend dashboard)
- **Pandas & Matplotlib** (data and plotting)

---

## 🧪 Running the App

### Prerequisites
- Python 3.9+
- [Gurobi installed and licensed](https://www.gurobi.com/downloads/gurobi-optimizer/)
- Flask

### Installation
```bash
pip install flask pandas matplotlib gurobipy
```

### Run the App
```bash
python app.py
```

Then open your browser to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Project Structure
```
├── app.py             # Flask app
├── model.py           # Gurobi optimisation model
├── templates/
│   └── index.html     # Frontend interface
├── static/
│   └── style.css      # CSS styles
├── data/
│   ├── conventional_datacentre.csv
│   └── underground_well_datacentre.csv
```

---

## 📦 Output
After running a simulation, the dashboard updates to show:
- Water savings (vs conventional)
- Energy consumption
- Estimated GPU temps
- Optimised flow rate

---

## 🤝 Contributors
Created by students committed to a sustainable future for AI infrastructure.

---

## 📄 License
MIT License

