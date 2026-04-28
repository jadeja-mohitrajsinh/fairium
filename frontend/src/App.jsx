import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import AnalysisWorkspace from "./pages/AnalysisWorkspace";
import BiasInsightsDashboard from "./pages/BiasInsightsDashboard";
import DecisionAnalysis from "./pages/DecisionAnalysis";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AnalysisWorkspace />} />
        <Route path="/dashboard" element={<BiasInsightsDashboard />} />
        <Route path="/decisions" element={<DecisionAnalysis />} />
      </Routes>
    </Router>
  );
}

export default App;
