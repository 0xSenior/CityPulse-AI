# CityPulse AI — Smart City Traffic, Safety & Environmental Analytics

> **Digital Egypt Pioneers Initiative (DEPI) — Round 5**  
> **Ministry of Communications and Information Technology (MCIT), Egypt**  
> **Training Partner:** Skills Dynamix  
> **Track:** AI & Data Science (Academic Year 2026 / 2027)

---

## 📌 Phase 1 Deliverables (Submission Due: 16 October 2026)

All required Phase 1 documentation and planning deliverables have been completed and organized:

| Deliverable | Description | Location Link |
| :--- | :--- | :--- |
| **Official Project Proposal** | Complete DEPI Proposal Template with team roles, objectives, tech stack, and 4-week milestones. | 📄 [Proposal_CityPulse_AI_Filled.docx](docs/Proposal_CityPulse_AI_Filled.docx) |
| **Planning & SRS Documentation** | Software Requirements Specification (SRS), Stakeholder Analysis, User Stories, and Risk Mitigation. | 📑 [DEPI_Project_Proposal_and_Planning.md](docs/DEPI_Project_Proposal_and_Planning.md) |
| **Literature Review** | Comprehensive state-of-the-art review comparing statistical, deep learning, and tree-based mobility models. | 📚 [Literature Review Section](docs/DEPI_Project_Proposal_and_Planning.md#2-literature-review--technical-background) |

---

## 👥 Team Members & Responsibilities

| Member | Role | Primary Responsibilities |
| :--- | :--- | :--- |
| **Ahmed (أحمد)** | **Project Team Leader & ML Engineer** | Overall project orchestration, model architecture, training, hyperparameter tuning, and serialization (XGBoost / Scikit-Learn). |
| **Mahdy (مهدي)** | **Traffic Data & Geospatial Pipeline Engineer** | Ingestion and preprocessing of automated traffic volume counts, coordinate transformation (`EPSG:2263` to `WGS84 Lat/Lon`), and feature engineering. |
| **Abdo (عبدو)** | **Accident Data & Safety Analytics Engineer** | Cleaning and analysis of NYC collision and transit records, extracting contributing factors, casualty metrics, and hotspot mapping. |
| **Sama (سما)** | **Environmental & Emissions Specialist** | Urban air quality parameterization ($NO_2, PM_{2.5}$), implementing EPA vehicle carbon footprint formulas ($CO_2$), and green routing analytics. |
| **Ziad (زياد)** | **Dashboard Developer & UI/UX Specialist** | Developing the interactive Streamlit web dashboard, UI/UX controls (borough, date, time filters), metric gauges, and geospatial maps. |
| **Basmala (بسملة)** | **System Analyst & Documentation Lead** | Software Requirements Specification (SRS), Stakeholder Matrix, User Stories, Literature Review, and UML / DFD architectural modeling. |

---

## 🚀 Project Overview

**CityPulse AI** is an intelligent urban intelligence platform designed to support traffic authorities and commuters by unifying three critical smart-city pillars from open municipal sensor data:

1. **Traffic Congestion Forecasting:** Predicts 15-minute interval vehicular volume counts across metropolitan corridors.
2. **Road Safety & Accident Hotspots:** Classifies collision risk tiers and identifies recurring accident risk factors.
3. **Environmental Impact & Emissions:** Estimates vehicular $CO_2$ emissions and monitors urban air quality trends ($NO_2, PM_{2.5}$).

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Data Engineering & Geospatial:** `pandas`, `numpy`, `pyproj`
* **Machine Learning & Modeling:** `scikit-learn`, `xgboost`, `joblib`
* **Interactive UI & Visualization:** `streamlit`, `folium`, `plotly`
* **Quality Assurance & Testing:** `pytest`
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
CityPulse-AI/
├── app/                  # Streamlit Interactive Web Application
├── data/                 # Processed datasets and street catalogs
├── docs/                 # Official DEPI Documentation & Proposal (.docx & .md)
│   ├── DEPI_Project_Proposal_and_Planning.md
│   └── Proposal_CityPulse_AI_Filled.docx
├── models/               # Serialized ML models and precomputed statistics
├── project/              # DEPI guideline references and proposal templates
├── src/                  # Core Python source code (data processor, modeling)
├── requirements.txt      # Project dependencies
└── README.md             # Repository documentation
```

---

## ⚡ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/0xSenior/CityPulse-AI.git
   cd CityPulse-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app/app.py
   ```
