# FairSight Core — Project Report

**Hackathon:** Google Solution Challenge 2026
**Track:** Unbiased AI Decision — Ensuring Fairness and Detecting Bias in Automated Decisions
**Team:** FairSight

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Technical Architecture](#technical-architecture)
5. [Features](#features)
6. [API Reference](#api-reference)
7. [Fairness Metrics](#fairness-metrics)
8. [Compliance Framework](#compliance-framework)
9. [Use Cases](#use-cases)
10. [Innovation & Differentiation](#innovation--differentiation)
11. [Impact](#impact)
12. [Future Roadmap](#future-roadmap)
13. [Appendix](#appendix)

---

## Executive Summary

FairSight Core is an end-to-end AI fairness auditing platform that enables organizations to detect, quantify, explain, and mitigate bias in automated decision-making systems. Built for the Google Solution Challenge 2026, it directly addresses one of the most pressing challenges in modern AI deployment: ensuring that algorithmic systems do not discriminate against individuals based on protected characteristics such as gender, race, or age.

The platform operates across three dimensions of bias analysis — **dataset bias**, **AI decision fairness**, and **text bias** — and provides actionable mitigation strategies alongside regulatory compliance assessments aligned with the EU AI Act and the US EEOC 80% Rule. FairSight is designed to be accessible to non-technical stakeholders while remaining rigorous enough for data scientists and compliance officers.

Key capabilities at a glance:

- Analyzes actual model predictions against real outcomes to detect decision-level discrimination
- Zero-configuration: auto-detects column roles, sensitive attributes, and target variables
- Computes 8 fairness metrics across dataset and decision levels
- Generates per-group confusion matrices with TP, FP, TN, FN, TPR, FPR, FNR, and precision
- Provides regulatory compliance verdicts (EU AI Act + EEOC 80% Rule)
- Offers active bias mitigation with downloadable debiased datasets
- Includes 4 real-world benchmark datasets for instant demonstration

---

## Problem Statement

AI systems now make or heavily influence life-changing decisions: who gets hired, who receives a loan, who is flagged as a recidivism risk, and who receives priority medical care. When these systems are trained on historically biased data, they do not merely reflect past discrimination — they institutionalize and scale it.

The consequences are severe and well-documented:

- **Hiring:** Resume screening tools have been shown to systematically downrank candidates from underrepresented groups.
- **Lending:** Credit scoring models have produced racially disparate approval rates in violation of the Equal Credit Opportunity Act.
- **Criminal Justice:** Risk assessment tools like COMPAS have assigned significantly higher recidivism scores to Black defendants than white defendants with similar profiles.
- **Healthcare:** Patient prioritization algorithms have been found to underestimate the care needs of Black patients relative to white patients with equivalent health conditions.

Despite the scale of this problem, most organizations lack accessible, practical tools to audit their AI systems for fairness. Existing academic tools require deep ML expertise, are not designed for compliance workflows, and do not address the full pipeline from detection through mitigation. FairSight was built to close this gap.

---

## Solution Overview

FairSight Core provides a unified platform for AI fairness auditing with three analysis modes and an active mitigation engine:

| Mode | Input | What It Detects |
|---|---|---|
| Dataset Bias Analysis | CSV with features + target | Structural bias in training data |
| AI Decision Fairness | CSV with predictions + outcomes + sensitive attributes | Discrimination in model outputs |
| Text Bias Analysis | Free text (job descriptions, policies, etc.) | Biased language by category |
| Active Mitigation | CSV dataset | Applies debiasing and returns corrected CSV |

The platform is built on a FastAPI backend with a React frontend, deployable via Docker in a single command. It is designed for three audiences simultaneously: data scientists who need rigorous metrics, compliance officers who need regulatory verdicts, and business stakeholders who need plain-language explanations.

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                  │
│  AnalysisWorkspace  │  BiasInsightsDashboard  │  DecisionAnalysis  │
└──────────────────────────────┬──────────────────────────┘
                               │ HTTP / REST
┌──────────────────────────────▼──────────────────────────┐
│                   FastAPI Backend (Python)                │
│                                                          │
│  /api/analyze          /api/analyze-decisions            │
│  /api/analyze-text     /api/detect-columns               │
│  /api/mitigate         /api/datasets                     │
└──────────────────────────────┬──────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼──────┐   ┌───────────▼──────┐   ┌──────────▼──────┐
│  Bias Engine  │   │  Decision Engine  │   │   Text Engine    │
│  (sklearn,    │   │  (6 fairness      │   │  (TF-IDF + LR +  │
│   pandas)     │   │   metrics, CM)    │   │   Gemini LLM)    │
└───────────────┘   └──────────────────┘   └─────────────────┘
```

### Backend File Structure

```
app/
├── main.py                          # FastAPI app, CORS, lifespan hooks
├── core/
│   ├── config.py                    # pydantic_settings, env vars, upload limits
│   ├── exceptions.py                # FairSightException, ValidationException, ServiceException
│   └── logging.py                   # Structured logging setup
├── api/
│   └── routes/
│       ├── __init__.py              # Combines all routers
│       ├── analysis.py              # /api/analyze, /api/analyze-text
│       ├── decisions.py             # /api/analyze-decisions, /api/detect-columns
│       ├── mitigation.py            # /api/mitigate
│       └── datasets.py              # /api/datasets, /api/datasets/{id}/download
├── schemas/
│   └── analysis.py                  # All Pydantic v2 response models
└── services/
    ├── bias/
    │   ├── fairness.py              # Main dataset bias analysis orchestrator
    │   ├── metrics.py               # DP difference, DI ratio, group selection rates
    │   ├── patterns.py              # Random Forest bias drivers, proxy detection
    │   └── decision_fairness.py     # Decision-level fairness (6 metrics, confusion matrix)
    ├── inference/
    │   ├── engine.py                # Auto column detection
    │   ├── data_loader.py           # CSV upload handling
    │   └── validator.py             # Input validation
    ├── mitigation/
    │   └── strategies.py            # Reweighting, feature removal, preprocessing recs
    ├── reporting/
    │   ├── insights.py              # Severity, confidence, explanations, structured report
    │   └── xai.py                   # SHAP importance, accuracy-fairness tradeoff
    └── ai/
        ├── gemini.py                # Google Gemini LLM integration
        └── text_bias.py             # Hybrid text bias analyzer
```

### Frontend File Structure

```
frontend/src/
├── App.jsx                          # Router: /, /dashboard, /decisions
├── api.js                           # All API calls (analyzeDataset, analyzeDecisions, etc.)
├── pages/
│   ├── AnalysisWorkspace.jsx        # Dataset upload, text analysis, sample datasets
│   ├── BiasInsightsDashboard.jsx    # Dataset results: metrics, charts, recommendations
│   └── DecisionAnalysis.jsx         # AI decision fairness: predictions, 6 metrics, compliance
├── components/
│   ├── FairnessCard.jsx             # Circular fairness score, group comparison
│   ├── ExecutiveSummary.jsx         # Compliance status, executive summary
│   ├── PriorityActions.jsx          # Urgent/monitor/safe action cards
│   ├── MitigationModal.jsx          # Debias dataset modal
│   └── TextAnalysisResult.jsx       # Text bias breakdown by category
└── styles.css                       # Full CSS (2500+ lines)
```

### Tech Stack

| Layer | Technologies |
|---|---|
| Backend framework | FastAPI, Python 3.11+ |
| Data validation | Pydantic v2 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| LLM integration | Google Generative AI (Gemini) |
| Frontend framework | React 18, Vite |
| Routing | React Router v6 |
| Charts | Recharts |
| Deployment | Docker, docker-compose |

---

## Features

### 1. AI Decision Fairness Analysis

The flagship feature of FairSight Core. Unlike tools that only analyze training data, FairSight analyzes **actual model predictions** against real outcomes to detect discrimination in what the model actually decided.

**How it works:**

1. Upload a CSV containing model predictions, actual outcomes, and sensitive attribute columns
2. FairSight auto-detects which column is the prediction, which is the ground truth, and which are sensitive attributes
3. The engine computes 6 decision-level fairness metrics across all demographic groups
4. A per-group confusion matrix is generated with full breakdown of TP, FP, TN, FN, TPR, FPR, FNR, and precision
5. An overall verdict is issued: **FAIR**, **POSSIBLY BIASED**, or **BIASED**
6. Regulatory compliance is assessed against EU AI Act requirements and the US EEOC 80% Rule
7. Concrete mitigation strategies are recommended: threshold calibration, post-processing equalized odds, group-specific threshold adjustment

**Endpoint:** `POST /api/analyze-decisions`
**Frontend:** `/decisions` page with column configuration UI, metric cards, and group statistics table

---

### 2. Dataset Bias Analysis

Analyzes structural bias in training datasets before a model is ever trained — catching problems at the source.

**Capabilities:**

- Upload any CSV with zero configuration required
- Auto-detects the target column and sensitive columns using keyword-based heuristics
- Computes Demographic Parity Difference and Disparate Impact Ratio
- Calculates per-group selection rates with statistical confidence scoring (HIGH / MEDIUM / LOW based on sample sizes)
- **Intersectional bias detection:** identifies bias in combinations of two or more sensitive attributes (e.g., Black women vs. white men)
- **Proxy bias detection:** uses correlation analysis to identify features that act as proxies for protected attributes even when those attributes are excluded
- **Affected population estimation:** quantifies how many individuals are disadvantaged by detected bias
- **Bias drivers:** uses Random Forest feature importance to identify which features most contribute to biased outcomes
- Severity classification: **LOW / MODERATE / HIGH**
- Structured compliance report with executive summary suitable for non-technical stakeholders
- Mitigation recommendations: reweighting, feature removal, preprocessing strategies

**Endpoint:** `POST /api/analyze`

---

### 3. Text Bias Analysis

Detects biased language in job descriptions, HR policies, loan application forms, and any free text that influences human or automated decisions.

**Detection categories:**

| Category | Examples |
|---|---|
| Gender bias | Gendered job titles, masculine-coded language |
| Racial bias | Racially coded language, cultural assumptions |
| Age bias | "Young and energetic," "digital native" |
| Socioeconomic bias | Class-coded language, wealth assumptions |
| Cultural bias | Cultural specificity presented as universal |
| Location bias | Geographic assumptions and stereotypes |

**Architecture:** Hybrid three-layer system:
1. **Rule-based layer:** Regex pattern matching against a curated library of biased phrases
2. **ML layer:** Logistic Regression classifier trained on TF-IDF features using labeled bias data
3. **LLM layer:** Google Gemini integration for nuanced contextual analysis (gracefully falls back if unavailable)

For each detected bias instance, FairSight suggests a neutral alternative phrasing. Confidence is reported as High, Medium, or Low.

**Endpoint:** `POST /api/analyze-text`

---

### 4. Active Bias Mitigation

Goes beyond detection to actively correct biased datasets and return a debiased CSV ready for retraining.

**Mitigation methods:**

| Method | Description | Best For |
|---|---|---|
| Reweighting (Kamiran/Calders) | Assigns sample weights to equalize group representation | When you cannot modify features |
| Disparate Impact Remover | Transforms feature distributions to reduce correlation with protected attributes | When feature modification is acceptable |
| Feature Removal | Removes identified proxy features | When proxies are clearly identified |
| Preprocessing Recommendations | Structured guidance for data collection improvements | Long-term data quality |

**Endpoint:** `POST /api/mitigate`

---

### 5. Sample Datasets

Four real-world fairness benchmark datasets are built into the platform for instant demonstration and testing, requiring no external downloads.

| Dataset | Rows | Domain | Key Fairness Question |
|---|---|---|---|
| Adult Income (Census) | 32,561 | Income prediction | Does the model predict income >$50K equally across races and genders? |
| IBM HR Employee Attrition | 1,470 | Hiring / HR | Does attrition prediction vary by gender or age? |
| German Credit Risk | 1,000 | Lending | Are loan approvals equally likely across demographic groups? |
| COMPAS Recidivism | 7,214 | Criminal justice | Are recidivism risk scores equally calibrated across racial groups? |

Each dataset can be loaded and analyzed with a single click from the Analysis Workspace.

**Endpoints:** `GET /api/datasets`, `GET /api/datasets/{id}/download`

---

### 6. SHAP Explainability

Provides model-level explainability using SHAP (SHapley Additive exPlanations) values computed via XGBoost, giving users a rigorous, game-theory-grounded understanding of which features drive biased outcomes.

- Trains an XGBoost model on the uploaded dataset
- Computes SHAP values for all features
- Visualizes feature importance as a horizontal bar chart in the dashboard
- Distinguishes between features that drive overall predictions and those that drive disparate outcomes
- Falls back gracefully if SHAP or XGBoost is not installed in the environment

---

### 7. Accuracy vs. Fairness Tradeoff Visualization

An interactive tool that makes the fundamental tension between predictive accuracy and fairness tangible and explorable.

- Interactive slider simulating mitigation levels from 0% (no intervention) to 100% (maximum fairness enforcement)
- Shows how accuracy and fairness scores change as mitigation intensity increases
- Helps decision-makers understand the cost of fairness interventions before committing to them
- Powered by simulation over the uploaded dataset, not theoretical curves

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns service status |
| `POST` | `/api/analyze` | Dataset bias analysis — upload CSV, returns full bias report |
| `POST` | `/api/analyze-text` | Text bias analysis — submit text, returns bias categories and alternatives |
| `POST` | `/api/analyze-decisions` | AI decision fairness — upload predictions CSV, returns 6 metrics + compliance |
| `POST` | `/api/detect-columns` | Auto-detect column roles in a CSV (prediction, outcome, sensitive) |
| `POST` | `/api/mitigate` | Apply bias mitigation — upload CSV, returns debiased CSV download |
| `GET` | `/api/datasets` | List all available sample datasets with metadata |
| `GET` | `/api/datasets/{id}/download` | Download a specific sample dataset by ID |

### Example: Analyze Decisions

```http
POST /api/analyze-decisions
Content-Type: multipart/form-data

file: predictions.csv
prediction_col: predicted_outcome    (optional — auto-detected if omitted)
actual_col: actual_outcome           (optional — auto-detected if omitted)
sensitive_cols: gender,race          (optional — auto-detected if omitted)
```

**Response (abbreviated):**

```json
{
  "verdict": "BIASED",
  "overall_fairness_score": 0.61,
  "metrics": {
    "demographic_parity": { "passed": false, "max_difference": 0.23 },
    "equalized_odds": { "passed": false, "tpr_difference": 0.18, "fpr_difference": 0.09 },
    "equal_opportunity": { "passed": false, "tpr_difference": 0.18 },
    "predictive_parity": { "passed": true, "precision_difference": 0.04 },
    "accuracy_parity": { "passed": true, "accuracy_difference": 0.03 },
    "fnr_parity": { "passed": false, "fnr_difference": 0.18 }
  },
  "group_stats": {
    "Male": { "tp": 412, "fp": 88, "tn": 310, "fn": 90, "tpr": 0.82, "fpr": 0.22, "fnr": 0.18, "precision": 0.82 },
    "Female": { "tp": 198, "fp": 52, "tn": 280, "fn": 170, "tpr": 0.54, "fpr": 0.16, "fnr": 0.46, "precision": 0.79 }
  },
  "compliance": {
    "eu_ai_act": "NON_COMPLIANT",
    "eeoc_80_rule": { "passed": false, "ratio": 0.66, "threshold": 0.80 }
  },
  "mitigation_recommendations": [
    "Apply threshold calibration per demographic group",
    "Post-processing equalized odds correction",
    "Group-specific threshold adjustment to equalize TPR"
  ]
}
```

---

## Fairness Metrics

### Dataset-Level Metrics

| Metric | Formula | Interpretation | Threshold |
|---|---|---|---|
| Demographic Parity Difference | P(Ŷ=1 \| A=a) − P(Ŷ=1 \| A=b) | Difference in selection rates between groups | < 0.1 = fair |
| Disparate Impact Ratio | P(Ŷ=1 \| A=a) / P(Ŷ=1 \| A=b) | Ratio of selection rates (EEOC 80% Rule basis) | > 0.8 = fair |

### Decision-Level Metrics

| Metric | What It Measures | Why It Matters |
|---|---|---|
| Demographic Parity | Equal selection rates across groups | Ensures proportional representation in positive decisions |
| Equalized Odds | Equal TPR and FPR across groups | Ensures errors are equally distributed — no group bears more false positives or false negatives |
| Equal Opportunity | Equal True Positive Rate (TPR) across groups | Qualified individuals have equal probability of receiving a positive decision regardless of group |
| Predictive Parity | Equal precision across groups | When the model predicts positive, it is equally likely to be correct for all groups |
| Accuracy Parity | Equal overall accuracy across groups | The model performs equally well for all groups |
| FNR Parity | Equal False Negative Rate across groups | Wrongful rejection rates are equal — no group is disproportionately denied |

### Verdict Logic

| Verdict | Condition |
|---|---|
| **FAIR** | 5–6 metrics pass |
| **POSSIBLY BIASED** | 3–4 metrics pass |
| **BIASED** | 0–2 metrics pass |

---

## Compliance Framework

FairSight maps its analysis results directly to two major regulatory frameworks:

### EU AI Act

High-risk AI systems (hiring, credit, education, law enforcement) are subject to mandatory bias testing under the EU AI Act. FairSight's decision fairness analysis produces a compliance status of:

- **COMPLIANT** — All critical fairness metrics pass
- **MONITOR** — Minor disparities detected; monitoring recommended
- **REQUIRES_ACTION** — Significant disparities; remediation required before deployment
- **NON_COMPLIANT** — Severe bias detected; system should not be deployed

### US EEOC 80% Rule (Four-Fifths Rule)

The EEOC's Uniform Guidelines on Employee Selection Procedures require that the selection rate for any protected group be at least 80% of the rate for the highest-selected group. FairSight computes this ratio directly from the uploaded predictions and reports a clear pass/fail with the computed ratio.

```
EEOC 80% Rule: selection_rate(minority) / selection_rate(majority) ≥ 0.80
```

---

## Use Cases

### 1. Hiring — Resume Screening Audit

A company uses an AI tool to screen resumes. An HR manager uploads a CSV of the tool's decisions (hired/not hired) alongside candidate gender and race data. FairSight computes Equal Opportunity scores and EEOC compliance, revealing that female candidates with equivalent qualifications are 28% less likely to receive a positive decision. The platform recommends threshold calibration and generates a compliance report for legal review.

### 2. Lending — Loan Approval Model

A bank's data science team uploads predictions from their credit scoring model alongside applicant race and income data. FairSight's Disparate Impact Ratio reveals that minority applicants are approved at 67% the rate of majority applicants — below the EEOC 80% threshold. The team uses the Active Mitigation feature to generate a reweighted training dataset and retrain the model.

### 3. Healthcare — Patient Prioritization

A hospital system audits its patient prioritization algorithm by uploading predicted care scores alongside actual care needs and patient demographics. FairSight's FNR Parity metric reveals that the model has a significantly higher false negative rate for elderly patients, meaning they are disproportionately denied priority care. The EU AI Act compliance report flags this as REQUIRES_ACTION.

### 4. Criminal Justice — Recidivism Tool Audit

A public defender's office loads the built-in COMPAS dataset and runs a full decision fairness analysis. The results reproduce the well-documented racial disparity in COMPAS scores, with the platform generating a structured report suitable for use in legal proceedings or policy advocacy.

### 5. HR Policy — Job Description Review

An HR team pastes a job description into the Text Bias Analyzer. FairSight identifies masculine-coded language ("competitive," "dominant," "ninja developer"), age-biased phrases ("young and energetic"), and suggests neutral alternatives for each, helping the team attract a more diverse applicant pool.

---

## Innovation & Differentiation

FairSight Core advances the state of the art in AI fairness tooling in seven key ways:

**1. Decision-level analysis, not just data-level**
Most fairness tools analyze training datasets. FairSight analyzes what the model actually decided — the predictions — against real outcomes. This is the only analysis that matters for regulatory compliance and real-world harm.

**2. Zero configuration**
FairSight auto-detects prediction columns, outcome columns, and sensitive attributes using keyword heuristics and statistical inference. Users upload a CSV and get results — no schema mapping, no column labeling, no ML expertise required.

**3. End-to-end pipeline**
FairSight covers the complete fairness workflow: detect bias → explain which features drive it → quantify how many people are affected → mitigate with a downloadable corrected dataset. No other tool in this space covers all four stages in a single platform.

**4. Accessible to non-technical users**
Every metric is explained in plain language. Compliance verdicts are stated clearly. Executive summaries are written for business stakeholders, not data scientists. The platform is designed to be used by HR managers, compliance officers, and legal teams — not just ML engineers.

**5. Real-world benchmark datasets built in**
Four of the most widely studied fairness datasets (Adult Income, IBM HR, German Credit, COMPAS) are available with one click, enabling immediate demonstration and benchmarking without any data preparation.

**6. Hybrid text analysis**
The text bias analyzer combines rule-based pattern matching, a trained ML classifier, and an LLM (Google Gemini) in a layered architecture. Each layer catches different types of bias, and the system degrades gracefully when the LLM is unavailable.

**7. Dual regulatory compliance**
FairSight is the only open-source fairness tool that simultaneously assesses compliance with both the EU AI Act and the US EEOC 80% Rule, making it immediately useful for organizations operating across jurisdictions.

---

## Impact

### Quantified Potential Impact

- **32,561 individuals** represented in the Adult Income benchmark dataset alone — each a real person whose financial opportunities were shaped by a biased model
- **7,214 individuals** in the COMPAS dataset — each affected by a risk score that influenced their liberty
- Organizations using FairSight can identify and correct bias before deployment, preventing discriminatory decisions at scale

### Alignment with UN Sustainable Development Goals

| SDG | Alignment |
|---|---|
| SDG 10: Reduced Inequalities | Directly reduces algorithmic discrimination against marginalized groups |
| SDG 16: Peace, Justice, Strong Institutions | Supports fair and accountable AI in criminal justice and public services |
| SDG 8: Decent Work and Economic Growth | Promotes fair hiring and equal economic opportunity |
| SDG 3: Good Health and Well-Being | Supports equitable healthcare resource allocation |

### Accessibility

FairSight is designed to democratize AI fairness auditing. By eliminating the need for ML expertise and providing plain-language outputs, it makes rigorous bias detection accessible to:

- Small organizations that cannot afford specialized AI ethics consultants
- Civil society organizations auditing public-sector AI systems
- Regulators and policymakers assessing AI deployments
- Journalists and researchers investigating algorithmic discrimination

---

## Future Roadmap

### Near-term (3–6 months)

- **Continuous monitoring:** Connect to live model endpoints and monitor fairness metrics over time with alerting
- **Audit trail:** Immutable log of all analyses for regulatory documentation
- **Additional fairness metrics:** Counterfactual fairness, individual fairness, calibration curves
- **PDF report export:** One-click generation of compliance reports for legal and regulatory submission

### Medium-term (6–12 months)

- **Model integration:** Direct integration with MLflow, Hugging Face, and Vertex AI for in-pipeline fairness checks
- **Multi-modal analysis:** Extend text bias detection to images and structured documents
- **Causal fairness analysis:** Move beyond correlation-based metrics to causal inference for more robust bias detection
- **Collaborative workflows:** Multi-user support with role-based access for data scientists, compliance officers, and executives

### Long-term (12+ months)

- **Regulatory intelligence:** Automated tracking of evolving AI fairness regulations across jurisdictions
- **Federated analysis:** Analyze bias in sensitive datasets without centralizing the data
- **Industry benchmarks:** Sector-specific fairness benchmarks for healthcare, finance, and criminal justice
- **API-first SaaS:** Hosted platform with usage-based pricing for enterprise customers

---

## Appendix

### Running the Project

**Prerequisites:** Docker and docker-compose installed

```bash
# Clone the repository
git clone [GitHub URL]
cd fairsight

# Start all services
./run_all.sh

# Or with Docker
docker-compose up --build
```

Services start at:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

### Environment Variables

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key    # Optional — enables LLM text analysis
MAX_UPLOAD_SIZE_MB=50
ALLOWED_ORIGINS=http://localhost:5173
```

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest tests/

# Run specific test modules
pytest tests/services/test_metrics.py
pytest tests/services/test_strategies.py
pytest tests/api/test_analysis.py
```

### Dependencies

```
fastapi
uvicorn
pydantic-settings
pandas
numpy
scikit-learn
xgboost
shap
google-generativeai
python-multipart
joblib
```

### Project Structure Summary

```
fairsight/
├── app/                    # FastAPI backend
├── frontend/               # React + Vite frontend
├── data/                   # Sample datasets and trained models
├── tests/                  # Test suite
├── docker/                 # Docker configuration
├── docker-compose.yml      # Multi-service orchestration
├── run_all.sh              # Start all services (Unix)
├── run_all.ps1             # Start all services (Windows)
├── requirements.txt        # Python dependencies
└── PROJECT_REPORT.md       # This document
```

---

*FairSight Core — Built for the Google Solution Challenge 2026*
*Making AI fairness auditing accessible to every organization.*
