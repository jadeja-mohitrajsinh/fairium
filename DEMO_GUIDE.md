# FairSight Core - Demo Walkthrough Guide

This guide provides a step-by-step walkthrough of FairSight Core's capabilities for the Google Solution Challenge judges.

---

##  Quick Start for Judges

### Option 1: Local Development (Recommended for Full Experience)

```bash
# Clone the repository
git clone <repository-url>
cd fairsight

# Start all services (Windows)
.\run_all.ps1

# Or on Linux/Mac
./run_all.sh
```

Access the application at:
- **Frontend**: http://127.0.0.1:5173
- **Backend API**: http://127.0.0.1:8001
- **API Documentation**: http://127.0.0.1:8001/docs

### Option 2: Docker Deployment

```bash
docker-compose up -d
```

---

##  Demo Scenarios

### Scenario 1: Dataset Bias Analysis (Hiring Data)

**Objective**: Detect gender bias in a hiring dataset

**Steps**:
1. Navigate to http://127.0.0.1:5173
2. Click on "Dataset Analysis" tab
3. Upload the sample dataset: `data/hr_attrition.csv`
4. Click "Analyze Dataset"
5. Review the result preview card showing:
   - Bias detected: Yes/No
   - Target column: Attrition
   - Sensitive attributes: Gender, Age, Job Level
   - Risk level
6. Click "View Full Insights" to see detailed dashboard

**Expected Results**:
- Fairness metrics for each sensitive attribute
- DP Difference and DI Ratio calculations
- Group selection rates with visual charts
- Bias drivers (top features influencing the target)
- Proxy features (features correlated with sensitive attributes)
- Impact assessment (how many individuals are disadvantaged)
- Mitigation recommendations (specific steps to fix bias)
- Before/after simulation showing potential improvement

**Key Talking Points**:
- "The system automatically detected gender as a sensitive attribute"
- "DP Difference of X% indicates Y level of disparity"
- "The simulation shows a Z% improvement with reweighting"

---

### Scenario 2: Text Bias Analysis (Job Description)

**Objective**: Detect bias in a job description

**Steps**:
1. Navigate to http://127.0.0.1:5173
2. Click on "Text Analysis" tab
3. Enter the following biased job description:
   ```
   We are looking for a young male candidate for this senior engineering position. 
   The ideal candidate should fit in with our traditional work culture and have a young mindset.
   ```
4. Click "Analyze Text"
5. Review the result showing:
   - Bias detected: Yes/Possible
   - Bias types: Gender, Age, Cultural
   - Confidence level
   - Suggested neutral alternatives

**Expected Results**:
- Detection of "male candidate" as gender bias
- Detection of "young" as age bias
- Detection of "traditional work culture" as cultural bias
- Neutral alternatives: "person" instead of "male/man", "all experience levels" instead of "young"
- ML confidence score from the hybrid classifier

**Key Talking Points**:
- "Our hybrid approach combines rule-based patterns with ML for accurate detection"
- "The system catches both obvious and subtle bias patterns"
- "Neutral alternatives help rewrite fair job descriptions"

---

### Scenario 3: Location Bias Detection

**Objective**: Detect geographic discrimination

**Steps**:
1. Navigate to Text Analysis tab
2. Enter:
   ```
   Do not hire people from rural backgrounds. We prefer urban candidates only.
   ```
3. Click "Analyze Text"

**Expected Results**:
- Bias detected: Yes
- Bias type: Location
- Explanation: "Do not hire people from rural" matches discriminatory pattern
- Suggested alternative: "candidates from all geographic backgrounds"

**Key Talking Points**:
- "The system detects location-based discrimination, which is often overlooked"
- "This is important for ensuring equal opportunity regardless of geographic origin"

---

##  Key Features to Highlight

### 1. Automatic Column Inference
- **What**: System automatically detects target and sensitive columns
- **Why**: No manual configuration needed - accessible to non-technical users
- **Demo**: Upload any CSV and watch it auto-detect columns

### 2. Hybrid Text Bias Detection
- **What**: Combines rule-based patterns with ML classification
- **Why**: Catches both obvious bias and subtle, context-dependent bias
- **Demo**: Test with "traditional work culture" (ambiguous) vs "male only" (obvious)

### 3. Plain Language Explanations
- **What**: All findings explained in business terms, not statistical jargon
- **Why**: Makes fairness accessible to HR, legal, and compliance teams
- **Demo**: Read the explanation sections - they're written for humans, not data scientists

### 4. Actionable Mitigation
- **What**: Specific recommendations to fix bias, not just detection
- **Why**: Organizations need to know HOW to fix problems
- **Demo**: Review the preprocessing steps and feature removal suggestions

### 5. Impact Quantification
- **What**: Estimates how many people are disadvantaged
- **Why**: Translates metrics into human impact
- **Demo**: Check the "Affected Population" section

### 6. Simulation
- **What**: Shows potential improvement before implementation
- **Why**: Builds confidence in mitigation strategies
- **Demo**: Review the simulation results showing % improvement

---

##  Impact & Social Good

### Real-World Applications
- **Hiring**: Ensure fair recruitment practices
- **Lending**: Comply with fair lending regulations
- **Healthcare**: Equitable medical AI systems
- **Criminal Justice**: Fair risk assessment tools
- **HR Policy**: Bias-free job descriptions

### SDG Alignment
- **SDG 5**: Gender Equality
- **SDG 8**: Decent Work and Economic Growth
- **SDG 10**: Reduced Inequalities
- **SDG 16**: Peace, Justice, and Strong Institutions

---

**Together, we can ensure AI benefits everyone equally.** 
