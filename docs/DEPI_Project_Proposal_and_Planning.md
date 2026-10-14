# Digital Egypt Pioneers Initiative (DEPI)
## Graduation Project Documentation: Phase 1 — Planning & Requirements
**Ministry of Communications and Information Technology (MCIT), Egypt**  
**Training Partner:** Dynamix Skills  
**Track:** AI & Data Science  
**Academic Year:** 2026 / 2027  

---

# Project Title: CityPulse AI  
### *Smart City Traffic Congestion & Incident Hotspot Forecasting with Knowledge-Grounded RAG Assistant*

---

## Document Control
| Item | Details |
| :--- | :--- |
| **Document Version** | 1.0 (Initial Submission) |
| **Submission Deadline** | 16 October 2026 |
| **Project Repository** | GitHub (Skills Dynamix Organization) |
| **Target Delivery Week** | Week 21 (Final Graduation Defense) |

---

## 1. Project Planning & Management

### 1.1 Project Proposal

#### 1.1.1 Overview & Background
Rapid urbanization in metropolitan cities has led to severe traffic congestion, increasing economic losses, elevated greenhouse gas emissions, and high accident vulnerability. Traditional navigation and traffic management systems rely heavily on historical averages or reactive, delayed notifications. 

**CityPulse AI** is an end-to-end intelligent Smart City solution engineered to proactively predict traffic bottlenecks and incident risk zones before they occur. By combining **predictive Machine Learning (XGBoost)** with **Generative AI and Retrieval-Augmented Generation (RAG)**, CityPulse AI enables municipal authorities, emergency responders, and everyday commuters to query road conditions in natural language (Arabic and English) and receive instantaneous, data-grounded forecasts, congestion severity assessments, and routing recommendations.

#### 1.1.2 Objectives
1. **Accurate Traffic Forecasting:** Predict 15-minute interval traffic volume (`Vol`) across over 14,000 road segments using historical sensor data and cyclic temporal features ($R^2 \ge 0.79$).
2. **Proactive Bottleneck & Incident Risk Detection:** Classify road segments into 4 congestion severity tiers and identify high-risk incident zones with $\ge 89\%$ accuracy.
3. **Conversational RAG Assistant:** Develop a natural language interface that semantically indexes road topologies, maps coordinates, and queries real-time ML predictions.
4. **Interactive Geospatial Dashboard:** Provide urban planners and drivers with a 3D geospatial map and interactive What-If simulation capabilities.

#### 1.1.3 Project Scope
* **In-Scope:**
  * Ingestion, geospatial transformation, and cleaning of multi-year automated traffic volume counts (NYC OpenData benchmark).
  * Feature engineering: temporal decomposition, rush-hour indicators, cyclic trigonometry encodings, and segment-level historical target encodings.
  * Training and serializing high-speed XGBoost regressors and classifiers.
  * Vector store indexing using ChromaDB and metadata catalogs for semantic street retrieval.
  * Natural language query routing (Intent Classifier for greetings, assistance, and domain queries).
  * Web application deployment using Streamlit and PyDeck.
* **Out-of-Scope:**
  * Real-time hardware IoT sensor deployment (simulated via high-frequency sensor dataset).
  * Autonomous vehicle steering control (focused purely on advisory and predictive analytics).

---

### 1.2 Task Assignment & Team Roles
According to DEPI project guidelines, the project team consists of 5 members with defined specializations:

| Member / Role | Primary Responsibilities | Deliverables Owned |
| :--- | :--- | :--- |
| **Member 1: Team Leader & ML Engineer** | Project orchestration, XGBoost model architecture, training, hyperparameter tuning, model serialization (`.joblib`). | `src/train_traffic_model.py`, `models/traffic_model.joblib`, ML metrics |
| **Member 2: Data & Geospatial Engineer** | Data pipeline, memory optimization, coordinate transformations (`EPSG:2263` to `WGS84`), street catalog generation. | `src/data_processor.py`, `data/processed/`, Geospatial schema |
| **Member 3: RAG & Backend Developer** | Vector database setup, ChromaDB indexing, semantic retrieval, prompt synthesis, Intent Classifier. | `src/vector_store.py`, `src/traffic_rag.py`, Query router |
| **Member 4: UI/UX & Frontend Developer** | Streamlit web application, PyDeck 3D geospatial map integration, What-If simulation interface. | `app/app.py`, UI/UX Wireframes, Design system |
| **Member 5: QA & Documentation Lead** | Unit testing, evaluation reports, GitHub repository governance, user manual, DEPI slide decks & infographics. | `test_rag_pipeline.py`, Documentation, DEPI Presentation |

---

### 1.3 Project Plan & Timeline (Gantt Chart)

The project timeline spans from Week 5 through Week 21, mapped directly to DEPI milestones:

```
[W5-W7] Planning & Proposal ──> [W8-W10] Requirements & Data ──> [W11-W14] Modeling & RAG ──> [W15-W18] App & QA ──> [W19-W21] Defense
```

| Phase | Weeks | Target Completion Date | Major Deliverables | Status |
| :--- | :---: | :---: | :--- | :---: |
| **Phase 1: Team Formation & Planning** | Weeks 5 – 7 | 16 Oct 2026 | Project Proposal, Team Allocation, Gantt Chart | **Completed** |
| **Phase 2: Requirements & Lit Review** | Weeks 8 – 10 | 16 Oct 2026 | SRS Document, Stakeholder Analysis, Use Cases | **Completed** |
| **Phase 3: System Analysis & Design** | Weeks 11 – 13 | 06 Nov 2026 | Architecture Diagrams, ERD, DFD, Sequence Diagrams | In Progress |
| **Phase 4: Implementation & Integration**| Weeks 14 – 17 | 30 Nov 2026 | XGBoost Models, ChromaDB Store, Streamlit App | **Implemented** |
| **Phase 5: Testing & Quality Assurance** | Weeks 18 – 19 | 20 Nov 2026 | Test Cases, Validation Metrics, Bug Reports | In Progress |
| **Phase 6: Final Presentation & Reports**| Weeks 20 – 21 | 04 Dec 2026 | Technical Report, DEPI Infographic, Defense Demo | Scheduled |

---

### 1.4 Risk Assessment & Mitigation Plan

| Risk ID | Risk Description | Severity | Probability | Mitigation Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **RSK-01** | **Memory Exhaustion on Large Dataset (~2GB CSV):** Loading raw records crashes memory on development machines. | High | High | Implement chunked streaming via Pandas/Polars and stratified subsampling (600k rows) preserving class balance. |
| **RSK-02** | **Irrelevant User Queries to RAG:** Users asking conversational questions ("Hello", "How are you?") receiving false traffic reports. | Medium | High | Integrate a deterministic **Query Intent Router** to intercept chitchat and guide the user before vector lookup. |
| **RSK-03** | **Unseen Street Names / Typographical Errors:** Users searching streets with non-standard spelling or Arabic transliteration. | Medium | Medium | Implement an Arabic-to-English semantic mapper, ordinal number cleaner, and fuzzy/substring phrase matching. |
| **RSK-04** | **Model Overfitting / Data Leakage:** Over-optimistic validation metrics due to segment-level memorization. | High | Medium | Execute strict 80/20 train/test splitting **prior** to computing target encodings and baseline segment statistics. |

---

### 1.5 Key Performance Indicators (KPIs)

#### Technical & Model KPIs:
* **Severe Incident & Congestion Classification Accuracy:** $\ge 85\%$ (Achieved: **`89.49%`**).
* **Severe Incident F1-Score:** $\ge 78\%$ (Achieved: **`80.72%`**).
* **Volume Regression Coefficient of Determination ($R^2$):** $\ge 0.70$ (Achieved: **`0.794`**).
* **Volume Mean Absolute Error (MAE):** $\le 45$ vehicles (Achieved: **`34.47` vehicles / 15 min**).

#### Operational & System KPIs:
* **End-to-End Inference Latency:** $< 1.0 \text{ second}$ per user query.
* **Knowledge Base Segment Coverage:** $100\%$ of indexed city corridors ($14,541$ unique segments).
* **System Availability:** $99.9\%$ uptime during demonstration and evaluation.

---

## 2. Literature Review & Technical Background

### 2.1 State-of-the-Art Review
1. **Classical Statistical Time-Series Models (ARIMA / SARIMAX):**
   * *Strengths:* Mathematically rigorous for single-sensor forecasting.
   * *Limitations:* Incapable of scaling across 14,000+ simultaneous road segments; fails to capture non-linear geospatial interactions.
2. **Deep Learning Architectures (LSTM / Spatial-Temporal GCN):**
   * *Strengths:* Captures temporal dependencies well.
   * *Limitations:* High computational cost, requires GPU clusters for training and inference, slow cold-start latency.
3. **Gradient Boosted Decision Trees (XGBoost / LightGBM) — *Selected Approach*:**
   * *Strengths:* Superior tabular performance, native support for missing values, `hist` tree method allows rapid training on CPU, low memory footprint, sub-millisecond inference.
4. **Retrieval-Augmented Generation (RAG) in Urban Mobility:**
   * Grounding conversational assistants with structured statistical catalogs prevents LLM hallucinations, ensuring commuters receive verified numerical volume and risk predictions.

---

## 3. Requirements Gathering (SRS)

### 3.1 Stakeholder Analysis
* **City Traffic Authorities & Municipal Engineers:** Need to identify emerging bottlenecks and accident-prone corridors to deploy traffic police or adjust signal timings.
* **Daily Commuters & Logistics Fleets:** Need proactive traffic advisories with specific congestion probabilities before embarking on trips.
* **DEPI Evaluation Committee:** Requires verifiable source code, rigorous data science methodology, clean documentation, and an intuitive UI.

---

### 3.2 User Stories & Use Cases

| User Story ID | Persona | As a... | I want to... | So that I can... |
| :---: | :---: | :--- | :--- | :--- |
| **US-01** | Commuter | Driver in Brooklyn | Ask in Arabic about congestion on Flatbush Ave at 6:00 PM | Decide whether to leave immediately or choose an alternative route. |
| **US-02** | Fleet Manager | Logistics Coordinator | Check incident risk on Washington Bridge during morning peak | Divert freight vehicles to prevent costly delivery delays. |
| **US-03** | Urban Planner | City Analyst | View an interactive 3D map of congested corridors by hour | Identify streets requiring infrastructure upgrades. |
| **US-04** | General User | First-time visitor | Greet the assistant and inquire about system capabilities | Understand how to interact with the platform easily. |

---

### 3.3 Functional Requirements (FR)

* **FR-01 (Data Preprocessing):** The system shall parse automated traffic volume logs, remove corrupted entries, and compute cyclic temporal encodings (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`).
* **FR-02 (Geospatial Mapping):** The system shall project local plane coordinates (`EPSG:2263`) into standard `WGS84` (Latitude / Longitude) for each street segment.
* **FR-03 (Target Encoding Lookup):** The system shall compute segment-level historical traffic statistics (`seg_mean`, `seg_hour_mean`, `seg_day_mean`, `seg_max`, `seg_std`) in an $O(1)$ fast lookup dictionary.
* **FR-04 (Predictive Inference):** The system shall provide real-time inference using serialized XGBoost models, returning estimated vehicle counts, congestion category, and incident probability.
* **FR-05 (Intent Routing):** The system shall classify incoming user queries into greetings, capability inquiries, gratitude, off-topic requests, or traffic inquiries.
* **FR-06 (Semantic Street Retrieval):** The system shall perform hybrid search (exact phrase matching + ChromaDB vector embeddings) to locate relevant road segments from user input.
* **FR-07 (Advisory Synthesis):** The system shall generate structured reports displaying alert icons (🟢 / 🟡 / 🔴), numerical metrics, and actionable travel advice.
* **FR-08 (Geospatial Visualization):** The web dashboard shall render an interactive 3D scatter map color-coding road segments based on congestion severity.

---

### 3.4 Non-Functional Requirements (NFR)

* **NFR-01 (Performance):** Prediction response time shall not exceed 1.0 second per query under local execution.
* **NFR-02 (Accuracy):** High-risk bottleneck classification accuracy shall remain $\ge 88\%$ on unseen test data.
* **NFR-03 (Usability & Accessibility):** The user interface shall support natural Arabic and English queries with high contrast, responsive UI/UX principles.
* **NFR-04 (Reliability & Robustness):** The system shall handle queries for unrecognized streets gracefully without unhandled exceptions or crashes.
* **NFR-05 (Portability):** The entire application stack shall be self-contained and reproducible using Python 3.10+ and standard dependencies specified in `requirements.txt`.
* **NFR-06 (Maintainability):** All source code must be modularized across dedicated packages (`src/`, `models/`, `app/`, `docs/`) with comprehensive docstrings and commit histories.

---

## 4. Next Steps & Phase 2 Roadmap
With Phase 1 (Planning, Literature Review, and Requirements Gathering) fully documented and the core Machine Learning / RAG pipeline operational, the team will proceed to:
1. **System Analysis & Design (Due: 06 Nov 2026):** Generating complete UML Diagrams (Use Case, Sequence, Activity, Class) and Data Flow Diagrams (DFD Level 0 & Level 1).
2. **Database Modeling:** Documenting the logical and physical schema for segment metadata and vector collections.
3. **Automated Testing Suite:** Writing formal PyTest suites covering feature transformations and model inference.

---
*Signed by Project Team & Submitted for Instructor Approval — DEPI 2026.*
