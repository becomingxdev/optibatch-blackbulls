<div align="center">
  <h1>🏭 OptiBatch</h1>
  <p><strong>AI-Driven Manufacturing Batch Optimization System</strong></p>
  
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
</div>

---

## 📖 Overview

**OptiBatch** is an advanced, AI-driven manufacturing software platform engineered to maximize batch process yields, enforce quality standards, and minimize energy consumption. Built for next-generation industrial environments, OptiBatch employs state-of-the-art machine learning models with a **stacked ensemble architecture**, maintaining an industry-leading prediction accuracy (target >92% based on strict time-aware cross-validation methodologies). 

With a robust **FastAPI backend** and an interactive **Streamlit Industrial Dashboard**, engineers can securely monitor live processes, execute Monte Carlo process simulations, and update "golden signatures" to establish new operational best practices.

---

## ✨ Key Features

### 🧠 Advanced Predictive Modeling
- **Performance Classification:** Machine learning pipelines capable of predicting critical batch outcomes and categorizing severity with high accuracy.
- **Process Simulation:** Execute Monte Carlo simulations to test theoretical batch parameter sweeps.

### ⚙️ Intelligent Golden Signature Optimization
- **Pareto Multi-Objective Optimization:** Balances trade-offs between yield, process time, and energy consumption.
- **Adaptive Weight Manager:** Dynamically tunes feature importance parameters in response to manufacturing shifts.

### 📊 Real-Time Process Monitoring & Anomaly Detection
- **Live Batch Tracking:** Continuous drift detection alerting operators to process deviations before quality drops.
- **Industrial Dashboard:** Real-time data visualization via an interactive Streamlit frontend. 

### 🔄 Continuous Learning Engine
- **Champion-Challenger Models:** Automatically evaluates and transitions newly trained challenger models that outperform current baseline champions.
- **Golden Signature Updates:** Constantly updates optimal execution paths as equipment efficiency changes.

### 🌿 Sustainability & Energy Analytics
- **Carbon Emission Calculator:** Real-time tracking of equivalent carbon emissions per batch.
- **Savings Projection:** Estimates ROI for specific optimization recommendations before they are physically executed.

---

## 🏗️ System Architecture

OptiBatch leverages a highly modular backend architecture designed for scalability and isolated testing:

- `api/`: REST APIs powered by FastAPI handling routing, optimization requests, and prediction serving.
- `models/`: Centralized store for ML model weights (`.pkl`) and model metadata.
- `optimization/`: Advanced optimization logic featuring Pareto engines and optimization objectives.
- `monitoring/`: Real-time batch comparison and alert generation.
- `continuous_learning/`: Automated retraining pipelines.
- `dashboard/`: A specialized visual interface constructed to bridge the API and shop-floor engineers.
- `frontend/`: (Optional) React & Vite web interface for extended enterprise controls.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js (if utilizing the React frontend)
- Virtual Environment tool (venv, conda)

### Backend & API Setup

1. **Navigate to the core directory**:
   ```bash
   cd backend/optibatch
   ```

2. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```

### Industrial Dashboard Setup

1. **Navigate to the Dashboard directory**:
   ```bash
   cd backend/dashboard
   ```

2. **Start the Streamlit application**:
   ```bash
   python scripts/run_api.py
   ```

---

## 📈 ML Methodology

Our machine learning evaluation rigorously follows the strict methodology used in top industrial benchmarks (e.g., the Blackbulls framework). We emphasize:
- **Time-Aware Cross-Validation**: Ensuring no future data leakage artificially inflates performance scores.
- **Stacked Ensembles**: Combining traditional gradient boosting models with Random Forests to maximize generalization mapping capabilities.
- **Interpretability**: Built-in SHAP (SHapley Additive exPlanations) reveal exactly *why* certain process variations contribute to specific yields.

---

## 🛡️ License & Acknowledgements

This property was developed to showcase an AI-driven Industry 4.0 ecosystem. It embodies best practices for enterprise structural integrity and high-accuracy machine learning deployment in physical manufacturing environments.

<div align="center">
  <i>Innovating the shop floor, one batch at a time.</i>
</div>
