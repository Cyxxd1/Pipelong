# Pipelong
## 🌍 Geothermal-Assisted AI Data Centre Cooling

### Project Summary
AI data centres consume large volumes of water for cooling, mostly through evaporative methods that waste water. As AI infrastructure grows, this problem scales with it.

**Pipelong** proposes a **closed-loop geothermal cooling system** that:
- Circulates just i.e., 15L of water to cool GPU chips
- Channels heated water to a i,e., 300L underground reservoir
- Passively cools using **natural underground geothermal conditions**
- Completely avoids evaporative cooling and unnecessary water waste

This approach improves **water efficiency**, reduces **energy costs**, and is **scalable** for modern AI infrastructure.

---

## 💻 App Overview

This repository includes a Flask web app that simulates and visualises underground cooling performance:

### ✨ Features:
- Interactive dashboard interface via browser
- Animated scroll effect introducing the concept
- Embedded iframe showing calculated outputs
- Parameter entry form with:
  - Energy and water usage
  - IT load, ambient temp, evaporation rate
  - Energy and water costs
- Displays real-time simulated results:
  - Underground vs Conventional comparison
  - Energy (kWh and J), water use, and total cost
  - Estimated cost savings

---

## 🌐 Web Frontend Details

- Modern UI with **scroll-based animation** using GSAP
- Slim **navbar with a large 100px logo** that overflows visually without stretching layout
- Clean **hero section** that communicates project goals
- Dashboard auto-generates results on form submission
- Responsive design with dark theme for high-tech feel

---

## 🚀 Future Expansion Ideas
- 🧠 Integrate **AI-based pre-cooling** prediction models
- 🔄 Add **dynamic flow rate adjustments** using ML
- 📈 Historical tracking or export of simulation results

---

## 💠 Tech Stack

| Layer        | Tech Used                        |
| ------------ | -------------------------------- |
| Backend      | Python (Flask), Gurobi (solver)  |
| Frontend     | HTML, CSS, JavaScript, GSAP      |
| Data/Visuals | Pandas, Matplotlib               |

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
Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure

```
├── app.py                  # Flask app and routing
├── model.py                # Gurobi optimisation logic
├── templates/
│   └── index.html          # Homepage with animation and iframe
│   └── dashboard.html      # Embedded dashboard view
├── static/
│   ├── styles.css     # Dashboard styles
│   └── images/             # Logo and background images
├── data/
│   ├── conventional_datacentre.csv
│   └── underground_well_datacentre.csv
```

---

## 📦 Output Summary

The app outputs:
- Simulated underground system performance
- Conventional system comparison
- Estimated annual **cost savings**
- Optimised energy + water usage based on user input

---

## 🤝 Contributors
Created by students working toward a smarter, more sustainable AI infrastructure ecosystem.

---

## 📄 License
MIT License

