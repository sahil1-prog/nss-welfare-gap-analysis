# Presentation Outline - NSS Challenge 5.2
## Welfare Scheme Participation & Gap Analysis

---

### Slide 1 - Title
- Project Title, Team Name, IIT Roorkee NSS
- Competition: NSS Open Projects 2026 - Challenge 5.2

---

### Slide 2 - The Problem
- India spends **₹3+ lakh crore** annually on welfare schemes
- Yet **30–50% eligible-but-not-enrolled gaps** persist (J-PAL estimates)
- Broad campaigns waste resources; **targeted outreach is 4× more cost-effective**
- Key Question: *Where exactly are the gaps, who is missing, and why?*

---

### Slide 3 - Our Approach
- Schemes Analysed: **PMUY** (clean cooking fuel) + **PM-KISAN** (farmer DBT)
- Data: Census 2011, NFHS-5, official scheme dashboards
- 3 composite indexes: **DVI** · **PGS** · **OPS**
- Machine Learning: K-Means clustering + Random Forest XAI
- Output: Policy Dashboard + Intervention Simulator

---

### Slide 4 - Data Architecture
- Harmonized 640+ districts across 4 datasets
- Boundary-matched using Census 2011 district codes
- Challenge: boundary changes post-2011 → parent-district rollup strategy

---

### Slide 5 - District Vulnerability Index (DVI)
- 6 indicators: literacy, SC/ST %, rurality, solid fuel use, banking, female literacy
- Normalized and equally weighted composite
- [Map: DVI choropleth - show UP/Bihar/Rajasthan cluster]

---

### Slide 6 - Participation Gap Score (PGS)
- Formula: (Eligible − Beneficiaries) / Eligible
- PMUY: **avg gap 34%** nationally; worst districts >65%
- PM-KISAN: **avg gap 28%**; worst in fragmented-landholding belts
- [Map: PGS choropleth - scheme selector]

---

### Slide 7 - District Clustering (4 Personas)
| Cluster | Label | Action |
|---|---|---|
| 0 | High Need, Low Coverage | Emergency outreach |
| 1 | High Need, Moderate Coverage | Intensify camps |
| 2 | Low Need, Low Coverage | Administrative fix |
| 3 | Low Need, High Coverage | Monitor & sustain |

---

### Slide 8 - Explainable AI Gap Diagnosis
- Random Forest on PGS ~ DVI features
- Top drivers: Female Literacy, Banking Penetration, SC/ST %
- Per-district "Why is this district lagging?" card
- [Screenshot: AI Recommendation Card]

---

### Slide 9 - Outreach Prioritization Map
- OPS = 0.4×PGS + 0.3×DVI + 0.3×log(Unserved Population)
- **Top 10 priority districts** shown on interactive map
- State-wise distribution of priority districts

---

### Slide 10 - Intervention Impact Simulator
- Interactive budget sliders: Awareness / Mobile Camps / Banking Drive
- Outputs: Expected PGS reduction, new beneficiaries, cost per enrolment
- [Screenshot: Simulator panel]

---

### Slide 11 - Policy Recommendations
1. **Cluster 0 districts** → Mobile enrolment camps + Gram Sabha drives
2. **Low female literacy districts** → Women-first DBT onboarding
3. **Low banking penetration** → BC Sakhi network expansion before PM-KISAN push
4. **Urban fringe gaps** → Digital help desks in CSCs

---

### Slide 12 - Impact & Feasibility
- Targeting top 50 districts could close **40% of the national gap** at <20% of broad campaign cost
- Methodology replicable across any DBT scheme
- Open-source, reproducible on GitHub

---

### Slide 13 - Live Dashboard Demo
- [Screenshot / Live Demo: app.py]
- Five panels: KPIs · Maps · Scorecards · Clusters · Simulator

---

### Slide 14 - Thank You & Q&A
- GitHub Repository Link
- Contact Details
