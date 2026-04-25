# Quick Start

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Start backend

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

## 3. Start frontend

```bash
cd frontend
npm install
npm run dev
```

## 4. Open the app

```text
http://127.0.0.1:5173
```

## 5. Use it

1. Upload a CSV file
2. Click `Run Bias Analysis`
3. Review:
   detected target
   detected sensitive columns
   fairness metrics
   bias drivers
   proxy features

## PowerShell shortcut

```powershell
.\run_all.ps1
.\stop_all.ps1
```
