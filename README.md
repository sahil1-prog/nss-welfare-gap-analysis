# 🇮🇳 Welfare Scheme Participation & Gap Analysis

### NSS IIT Roorkee Open Projects 2026 - Challenge 5.2

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)

---

## 📌 Overview

This project analyses **participation gaps** in two major Indian welfare schemes -  
**Pradhan Mantri Ujjwala Yojana (PMUY)** and **Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)** -  
across 640+ districts of India using publicly available datasets (Census 2011, NFHS-5, official scheme dashboards).

It builds three composite policy indexes, a district clustering model, an Explainable AI (XAI)  
gap-driver analysis, and a fully interactive Streamlit policy dashboard with an Intervention Impact Simulator.

---

## 🎯 Key Features

| Feature                                 | Description                                                             |
| --------------------------------------- | ----------------------------------------------------------------------- |
| **District Vulnerability Index (DVI)**  | Composite marginalization score from 6 Census/NFHS-5 indicators         |
| **Participation Gap Score (PGS)**       | % of eligible population not covered by each scheme                     |
| **Outreach Prioritization Score (OPS)** | Weighted blend of gap severity, vulnerability & unserved population     |
| **District Clustering**                 | K-Means segmentation into policy personas                               |
| **Explainable AI Recommendations**      | Random Forest + SHAP-style feature importance for gap diagnosis         |
| **Intervention Impact Simulator**       | Budget-aware simulation of policy levers with cost-effectiveness ratios |
| **Interactive Choropleth Maps**         | Folium + Plotly maps for DVI / PGS / OPS at district level              |

---

## 🗂️ Project Structure

```
nss_52/
├── data/
│   ├── raw/                   # Original source CSVs
│   ├── processed/             # Harmonized merged dataset
│   └── geojson/               # India district boundaries
├── notebooks/
│   └── analysis.ipynb         # Full EDA → Indexes → ML → Recommendations
├── src/
│   ├── data_loader.py         # Loading, cleaning, merging pipeline
│   ├── index_builder.py       # DVI, PGS, OPS calculations
│   ├── models.py              # Clustering + Random Forest XAI
│   └── recommender.py         # Policy recommendation engine
├── assets/                    # Static images for dashboard
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
├── presentation_outline.md    # Judge/mentor presentation structure
└── final_report.md            # Academic policy report template
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/sahil1-prog/nss-welfare-gap-analysis.git
cd nss-welfare-gap-analysis

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate / download data
python src/data_loader.py

# 5. Run the interactive dashboard
streamlit run app.py
```

---

## 📊 Data Sources

| Dataset          | Source                                                    | Variables Used                       |
| ---------------- | --------------------------------------------------------- | ------------------------------------ |
| Census 2011 PCA  | [censusindia.gov.in](https://censusindia.gov.in) / GitHub | Population, Literacy, SC/ST, Workers |
| NFHS-5 (2019-21) | [rchiips.org/nfhs](http://rchiips.org/nfhs/) / GitHub     | Cooking fuel, Banking, Electricity   |
| PMUY Uptake      | MoPNG / Synthetic (calibrated)                            | LPG connections per district         |
| PM-KISAN Uptake  | MoAFW / Synthetic (calibrated)                            | Active beneficiaries per district    |
| District GeoJSON | Datameet / OpenDataMaps                                   | Boundary polygons for mapping        |

> **Note:** Scheme-level district uptake data is synthesized using officially reported national totals  
> and state-level distributions to maintain statistical validity where raw district CSVs are unavailable.

---

## 🧠 Methodology

```
Raw Data → Harmonization → Feature Engineering → EDA → Spatial Analysis
       → Clustering → XAI Gap-Driver Model → Recommendation Engine → Dashboard
```

1. **Data Harmonization:** All datasets are joined on 2011 Census district codes.
2. **Index Engineering:** DVI, PGS, OPS computed per district (see `src/index_builder.py`).
3. **Clustering:** K-Means (k=4) segments districts into policy personas.
4. **XAI:** Random Forest regressor on PGS; top features drive automated recommendations.
5. **Dashboard:** Streamlit app with Folium maps and Plotly interactive charts.

---

## 📈 Key Findings (Sample)

- Over **38% of eligible rural households** in bottom-quartile districts remain uncovered by PMUY.
- Districts with low female literacy (<40%) show **2.3× higher PMUY participation gaps**.
- PM-KISAN coverage gaps are most severe in **high-fragmented landholding** districts of UP, Bihar, and MP.
- The top 10 priority districts for outreach are concentrated in **4 states** (UP, Bihar, Rajasthan, MP).

---

## 📝 Reproducibility

All random seeds are fixed (`RANDOM_STATE = 42`). The data generation pipeline is deterministic.  
Run `python src/data_loader.py` to regenerate all processed datasets from scratch.

---

## 👥 Authors

- **Student:** Sahil, NSS IIT Roorkee
- **Competition:** NSS Open Projects 2026 - Challenge 5.2
