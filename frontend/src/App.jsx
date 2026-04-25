import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import AnalysisWorkspace from "./pages/AnalysisWorkspace";
import BiasInsightsDashboard from "./pages/BiasInsightsDashboard";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AnalysisWorkspace />} />
        <Route path="/dashboard" element={<BiasInsightsDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
