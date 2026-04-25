# FairSight Core

**[Google Solution Challenge 2026] Unbiased AI Decision - Ensuring Fairness and Detecting Bias in Automated Decisions**

Computer programs now make life-changing decisions about who gets a job, a bank loan, or even medical care. However, if these programs learn from flawed or unfair historical data, they will repeat and amplify those exact same discriminatory mistakes.

FairSight Core is a comprehensive bias analysis and mitigation platform that helps organizations **measure, flag, and fix harmful bias before their systems impact real people**.

---

## 🎯 Problem Statement

AI systems increasingly make critical decisions affecting human lives. When trained on biased historical data, these systems perpetuate and amplify discrimination against protected groups. Organizations lack accessible tools to detect and mitigate algorithmic bias before deployment.

**Our Solution**: FairSight Core provides an end-to-end platform for detecting, analyzing, and mitigating bias in datasets and AI systems, making fairness accessible to organizations of all sizes.

---

## ✨ Key Features

### 🔍 Comprehensive Bias Detection
- **Automatic column inference** - Intelligently detects target and sensitive columns from your data
- **Fairness metrics** - Demographic Parity Difference, Disparate Impact Ratio, and more
- **Severity classification** - LOW/MODERATE/HIGH risk levels with color-coded indicators
- **Intersectional bias analysis** - Detects bias across combinations of sensitive attributes
- **Proxy bias detection** - Identifies features that may indirectly encode sensitive attributes
- **Text bias analysis** - Hybrid ML + rule-based detection of bias in text (job descriptions, policies, etc.)

### 📊 Human-Readable Insights
- **Plain language explanations** - Clear, actionable descriptions of bias findings
- **Sample size awareness** - Reliability markers based on group sample sizes
- **Confidence scoring** - HIGH/MEDIUM/LOW confidence based on data quality
- **Distribution context** - Avoids misleading conclusions with population trends
- **Visual dashboards** - Interactive charts and graphs for intuitive understanding

### 🛡️ Impact Assessment
- **Affected population estimation** - Quantifies how many individuals are disadvantaged
- **Real-world impact** - Translates metrics into human impact
- **Group-level analysis** - Detailed breakdown by demographic groups
- **Compliance status** - COMPLIANT/MONITOR/REQUIRES_ACTION classification

### 🔧 Bias Mitigation
- **Data preprocessing recommendations** - Specific steps to reduce bias in training data
- **Feature removal suggestions** - Identifies proxy features to remove or transform
- **Reweighting strategies** - Sample weight calculations for balanced representation
- **Before/after simulation** - Shows potential improvement with balanced outcomes
- **Neutral alternatives** - Suggests fair language for text bias

### 📋 Compliance & Reporting
- **Bias report summary** - Executive-level overview for compliance officers
- **Professional audit format** - Clear, trustworthy reporting for stakeholders
- **Export capabilities** - Generate reports for regulatory submissions
- **Actionable recommendations** - Specific guidance per severity level

## 🚀 Quick Start

### One-command start

Windows PowerShell:

```powershell
.\run_all.ps1
.\stop_all.ps1
```

Linux/Mac:

```bash
./run_all.sh
./stop_all.sh
```

The application will be available at:
- **Frontend**: http://127.0.0.1:5173
- **Backend API**: http://127.0.0.1:8001
- **API Docs**: http://127.0.0.1:8001/docs

### Manual Setup

**Backend:**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```text
fairsight/
├── app/
│   ├── api/
│   ├── models/
│   ├── services/
│   └── main.py
├── data/
├── frontend/
├── requirements.txt
├── run_all.ps1
└── stop_all.ps1
```

## Backend

Run the API directly:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Health endpoint:

```text
http://127.0.0.1:8001/health
```

Swagger docs:

```text
http://127.0.0.1:8001/docs
```

## Frontend

Run the frontend directly:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on:

```text
http://127.0.0.1:5173
```

## One-command start

Windows PowerShell:

```powershell
.\run_all.ps1
.\stop_all.ps1
```

## 📖 Use Cases

### Hiring & Recruitment
Ensure your hiring algorithms don't discriminate based on gender, race, age, or other protected characteristics. FairSight helps you identify and mitigate bias before it affects real candidates.

### Lending & Credit
Detect unfair lending practices that may disproportionately deny loans to certain demographic groups. Comply with fair lending regulations and build trust with customers.

### Healthcare & Medical AI
Ensure medical AI systems provide equitable care across all patient populations. Identify disparities in diagnosis, treatment recommendations, or resource allocation.

### Criminal Justice & Risk Assessment
Audit risk assessment tools for racial or socioeconomic bias. Ensure fair treatment across all communities.

## 📚 API Documentation

### `POST /analyze`

Upload a CSV file for comprehensive bias analysis.

**Request:**
- Multipart form data with `file` field containing the CSV

**Response:** Comprehensive bias analysis including:

```json
{
  "summary": "Executive summary of findings",
  "detected_target": "auto-detected target column",
  "detected_sensitive_columns": ["gender", "race", "age_group"],
  "potential_bias_detected": true,
  "fairness_metrics": {
    "gender": {
      "dp_diff": 0.154,
      "di_ratio": 0.395,
      "group_rates": {"male": 0.78, "female": 0.56},
      "explanation": "Plain language explanation",
      "severity": "HIGH",
      "group_analysis": [
        {
          "group": "male",
          "total_samples": 500,
          "positive_outcomes": 390,
          "selection_rate": 0.78,
          "reliability": "HIGH"
        }
      ],
      "confidence": "HIGH",
      "confidence_explanation": "Confidence assessment",
      "recommendation": "Actionable guidance",
      "simulation": {
        "dp_diff_reduced": 0.08,
        "improvement": "48%",
        "explanation": "Simulation explanation"
      }
    }
  },
  "bias_drivers": [
    {"feature": "income", "impact": 0.32}
  ],
  "proxy_features": [
    {
      "sensitive_column": "gender",
      "feature": "zipcode",
      "correlation": 0.65,
      "explanation": "Human-readable proxy explanation"
    }
  ],
  "intersectional_bias": [
    {
      "group": "Female + Single",
      "selection_rate": 0.28,
      "risk_level": "HIGH"
    }
  ],
  "notes": ["Contextual notes about distribution"],
  "affected_population": {
    "gender": {
      "total_affected_individuals": 45,
      "affected_groups": [...],
      "explanation": "Impact assessment"
    }
  },
  "preprocessing_steps": [
    {
      "type": "reweighting",
      "column": "gender",
      "issue": "High disparity detected",
      "recommendation": "Apply sample reweighting",
      "priority": "HIGH"
    }
  ],
  "feature_removals": [
    {
      "feature": "zipcode",
      "sensitive_column": "race",
      "correlation": 0.72,
      "rationale": "High correlation with sensitive attribute",
      "priority": "HIGH"
    }
  ],
  "bias_report_summary": {
    "overall_risk_level": "HIGH",
    "total_sensitive_attributes_analyzed": 3,
    "total_records_analyzed": 1000,
    "high_risk_attributes": ["gender"],
    "moderate_risk_attributes": ["age_group"],
    "low_risk_attributes": ["race"],
    "compliance_status": "REQUIRES_ACTION",
    "recommendation_summary": "Executive summary"
  }
}
```

## 🏗️ Architecture

FairSight Core uses a modern, scalable architecture:

- **Backend**: FastAPI (Python) - High-performance API with automatic validation
- **Frontend**: React + Vite + React Router - Fast, modern SPA with page navigation
- **Analysis Engine**: Pandas + NumPy + Scikit-learn - Efficient statistical computations
- **Text Bias Detection**: Hybrid ML (Logistic Regression + TF-IDF) + Rule-based patterns
- **Fairness Metrics**: Industry-standard measures (DP, DI, etc.)
- **Visualization**: Recharts - Interactive data visualizations

### Technology Stack

```
Backend:
├── FastAPI (Web Framework)
├── Pydantic (Data Validation)
├── Pandas (Data Processing)
├── NumPy (Numerical Computing)
├── Scikit-learn (ML for text bias)
└── River (Online Learning)

Frontend:
├── React 18 (UI Framework)
├── Vite (Build Tool)
├── React Router (Navigation)
├── Recharts (Data Visualization)
└── Modern CSS (Styling)
```

## 🎨 User Interface

FairSight Core features a modern, professional SaaS-style interface:

### Page 1: Analysis Workspace
- **Tabbed navigation** - Switch between Dataset Analysis and Text Analysis
- **Drag-and-drop CSV upload** - Easy file selection with preview
- **Text input area** - Analyze job descriptions, policies, and other text
- **Result preview cards** - Quick summary with bias status, type, and confidence
- **CTA buttons** - "View Full Insights" to navigate to detailed dashboard

### Page 2: Bias Insights Dashboard
- **Summary banner** - Risk level, records analyzed, key metrics at a glance
- **Card-based layout** - Each sensitive attribute in its own card
- **Interactive charts** - Bar charts for group selection rates
- **Collapsible sections** - Bias Drivers, Intersectional Bias, Impact Assessment, Recommendations
- **Color-coded indicators** - Red (high risk), Yellow (moderate), Green (safe)

## 📖 Use Cases

### Hiring & Recruitment
Ensure your hiring algorithms don't discriminate based on gender, race, age, or other protected characteristics. FairSight helps you identify and mitigate bias before it affects real candidates.

### Lending & Credit
Detect unfair lending practices that may disproportionately deny loans to certain demographic groups. Comply with fair lending regulations and build trust with customers.

### Healthcare & Medical AI
Ensure medical AI systems provide equitable care across all patient populations. Identify disparities in diagnosis, treatment recommendations, or resource allocation.

### Criminal Justice & Risk Assessment
Audit risk assessment tools for racial or socioeconomic bias. Ensure fair treatment across all communities.

### HR Policy Review
Analyze job descriptions, company policies, and internal communications for subtle bias using our hybrid text analysis system.

## 📚 API Documentation

### `POST /analyze`
Upload a CSV file for comprehensive bias analysis.

**Request:**
- Multipart form data with `file` field containing the CSV

**Response:** Comprehensive bias analysis including fairness metrics, bias drivers, proxy features, and recommendations.

### `POST /analyze-text`
Analyze text for potential bias using hybrid ML + rule-based detection.

**Request:**
```json
{
  "text": "Your text to analyze"
}
```

**Response:**
```json
{
  "bias_detected": "Yes/Possible/No",
  "biases": [
    {
      "type": "gender/race/age/location/cultural",
      "confidence": "High/Medium/Low",
      "explanation": "Detailed explanation",
      "alternatives": ["Suggested neutral alternatives"]
    }
  ],
  "overall_confidence": "High/Medium/Low",
  "ml_confidence": 0.85,
  "summary": "Summary of findings"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### Interactive API Docs
Swagger UI available at: `http://127.0.0.1:8001/docs`

## 🚀 Deployment

### Docker Deployment (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Stop services
docker-compose down
```

### Manual Deployment

**Backend:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

**Frontend:**
```bash
cd frontend
npm install
npm run build
# Serve the dist/ folder with any web server
```

## 🧪 Testing

Run the test suite:

```bash
# Backend tests
pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📊 Sample Datasets

The `data/` directory contains sample datasets for testing:

- `hr_attrition.csv` - HR attrition data with gender, age, job level
- `compas-scores-raw.csv` - Criminal justice risk assessment data
- `german_credit_data.csv` - Credit scoring dataset

## 🎓 How It Works

### Dataset Bias Analysis Pipeline

1. **Upload CSV** → User uploads dataset via drag-and-drop
2. **Column Inference** → System auto-detects target and sensitive columns
3. **Fairness Metrics** → Calculates DP, DI, and other metrics per attribute
4. **Bias Detection** → Identifies bias drivers, proxy features, intersectional bias
5. **Impact Assessment** → Quantifies affected population
6. **Mitigation Recommendations** → Suggests preprocessing steps and feature removals
7. **Simulation** → Shows potential improvement with balanced outcomes

### Text Bias Analysis Pipeline

1. **Text Input** → User enters text (job description, policy, etc.)
2. **Rule-Based Detection** → Matches against bias patterns (gender, race, age, location, cultural)
3. **ML Classification** → Logistic Regression classifier predicts bias probability
4. **Hybrid Scoring** → Combines rule-based and ML results for final assessment
5. **Neutral Alternatives** → Suggests fair language replacements

## 🏆 Competitive Advantages

### 1. Accessibility
- **No-code interface** - Non-technical users can detect bias without ML expertise
- **Plain language** - Explanations in business terms, not statistical jargon
- **Visual feedback** - Color-coded risk levels and interactive charts

### 2. Comprehensive Coverage
- **Dataset + Text** - Analyzes both structured data and unstructured text
- **Hybrid approach** - Combines rule-based patterns with ML for accuracy
- **Intersectional analysis** - Detects bias across combined attributes

### 3. Actionable Insights
- **Specific recommendations** - Not just "bias detected" but "here's how to fix it"
- **Simulation** - Shows potential improvement before implementation
- **Prioritization** - Urgent vs. monitor vs. safe categorization

### 4. Enterprise-Ready
- **Compliance reporting** - Professional format for regulatory submissions
- **Scalable architecture** - FastAPI + React for performance
- **Docker deployment** - Easy containerization and cloud deployment

## 🌟 Impact & Social Good

By providing accessible bias detection and mitigation tools, FairSight Core helps:

- **Organizations** build fairer AI systems and comply with regulations
- **Job seekers** receive fair consideration regardless of demographic factors
- **Loan applicants** get equitable access to financial services
- **Patients** receive unbiased medical care recommendations
- **Communities** affected by historical discrimination
- **Regulators** enforce fairness standards with better tools
- **Researchers** advance fairness in AI with open-source tools

**SDG Alignment**: This project contributes to UN Sustainable Development Goals:
- **SDG 5**: Gender Equality
- **SDG 8**: Decent Work and Economic Growth
- **SDG 10**: Reduced Inequalities
- **SDG 16**: Peace, Justice, and Strong Institutions

## 🤝 Contributing

FairSight Core is open source and welcomes contributions. Areas for improvement:
- Additional fairness metrics (Equal Opportunity, Equalized Odds)
- Model-level bias analysis (beyond dataset-level)
- Integration with ML pipelines (TensorFlow, PyTorch)
- Export to compliance report formats (PDF, DOCX)
- Multi-language support for text bias detection
- Real-time collaboration features
- Advanced visualization options

## 📄 License

This project is part of the Google Solution Challenge 2026.

## 🙏 Acknowledgments

- Google Solution Challenge 2026 for the inspiration
- Fairness research community for foundational metrics
- Open-source libraries that power this solution

---

**Together, we can ensure AI benefits everyone equally.** 🌍
# fairium
# fairium
