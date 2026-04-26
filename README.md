# earthquake-dashboard
# 🌍 DOTEKHA — Earthquake Vulnerability Dashboard

> An interactive Agent-Based Model (ABM) simulation and analytical dashboard for exploring socioeconomic vulnerability to earthquakes, grounded in the **Wisner / Pressure and Release (PAR) Framework**.

[![Streamlit App](https://earthquake-dashboard-mth4r8qvrweuutwe5nkc74.streamlit.app/)

---

## Overview

DOTEKHA simulates how earthquake vulnerability varies across **three socioeconomic groups** using an Agent-Based Model. Each agent represents an urban resident whose vulnerability is computed through a theory-driven neural architecture inspired by the PAR model.

The dashboard transforms raw simulation data into a fully interactive research tool — allowing researchers to filter by scenario, time step, and individual agent, and to explore vulnerability patterns across economic, social, physical, access, and household dimensions.

---

## Research Context

This project is grounded in **Wisner et al.'s Pressure and Release (PAR) model**, which frames disaster vulnerability as the intersection of hazard exposure and social vulnerability. The ABM operationalises this by encoding PAR dimensions as hidden nodes in a weighted feedforward network.

### Socioeconomic Scenarios

| Scenario | Description | Mean Vulnerability |
|---|---|---|
|  High SES | Resilient — high income, education, transport access | ~0.609 |
|  Middle SES | Mixed — moderate resources and risk factors | ~0.636 |
|  Low SES | Highly Vulnerable — low income, poor access, high health burden | ~0.666 |

---

##  Model Architecture

### Agent Inputs (x1–x6)
| Variable | Description |
|---|---|
| `income_level` | Household income (normalised 0–1) |
| `education_level` | Educational attainment |
| `age` | Age (higher = more vulnerable) |
| `health_limitation` | Physical health burden |
| `transport_access` | Access to transport infrastructure |
| `household_size` | Number of dependents |

### Hidden Layer — PAR Dimensions (h1–h5)
| Node | Dimension |
|---|---|
| `h1_economic` | Economic vulnerability |
| `h2_social` | Social vulnerability |
| `h3_physical` | Physical vulnerability |
| `h4_access` | Access vulnerability |
| `h5_household` | Household vulnerability |

### Output
- `vulnerability` — Final score (0–1), computed via sigmoid activation

---

##  Dashboard Features

| Tab | Contents |
|---|---|
| **Overview** | Summary metrics, scenario comparison cards, mean vulnerability bar chart |
| **Simulation** | Vulnerability trend over 50 time steps, distribution histogram, boxplot |
| **Hidden Layer** | PAR dimension activations grouped by scenario, radar charts |
| **Agent Explorer** | Per-agent inputs, hidden values, radar chart, and vulnerability trajectory |
| **Prediction** | R² and RMSE for Neural Network, Random Forest, XGBoost; feature importance |
| **Correlation** | Full correlation heatmap (inputs + hidden nodes + vulnerability) |

All charts are built with **Plotly** for full interactivity. Every chart responds dynamically to the sidebar filters.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dotekha-dashboard.git
cd dotekha-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

macOS users: XGBoost requires OpenMP. Run `brew install libomp` if you see a `libxgboost.dylib` error.

### 3. Run the dashboard
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. If the CSV files are missing, the ABM regenerates the data automatically.


##  File Structure

```
dotekha-dashboard/
│
├── app.py                    # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── abm_data_full.csv         # Full simulation data (500 agents × 50 steps × 3 scenarios)
├── abm_data_final_step.csv   # Final step snapshot (used for model training)
├── dotekha.ipynb             # Original Colab notebook (model development)
└── README.md                 # This file
```

---

Requirements

streamlit
plotly
pandas
numpy
scikit-learn
xgboost


##  Live Demo

The dashboard is deployed on **Streamlit Community Cloud**:

🔗 [https://earthquake-dashboard-mth4r8qvrweuutwe5nkc74.streamlit.app/](https://earthquake-dashboard-mth4r8qvrweuutwe5nkc74.streamlit.app)

