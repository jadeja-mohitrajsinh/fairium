import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";

import AnalysisWorkspace from "./pages/AnalysisWorkspace";
import BiasInsightsDashboard from "./pages/BiasInsightsDashboard";
import DecisionAnalysis from "./pages/DecisionAnalysis";
import XAIExplainer from "./pages/XAIExplainer";
import Login from "./pages/Login";

function GlobalHeader() {
  const { user, logout } = useAuth();
  const location = useLocation();

  if (!user || location.pathname === "/login") return null;

  return (
    <div style={{ position: "absolute", top: "1rem", right: "1rem", zIndex: 1000 }}>
      <button 
        onClick={logout}
        style={{
          background: "transparent",
          color: "white",
          border: "1px solid rgba(255,255,255,0.3)",
          padding: "0.5rem 1rem",
          borderRadius: "4px",
          cursor: "pointer",
          fontSize: "0.9rem"
        }}
      >
        Logout
      </button>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <GlobalHeader />
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<ProtectedRoute><AnalysisWorkspace /></ProtectedRoute>} />
          <Route path="/dashboard" element={<ProtectedRoute><BiasInsightsDashboard /></ProtectedRoute>} />
          <Route path="/decisions" element={<ProtectedRoute><DecisionAnalysis /></ProtectedRoute>} />
          <Route path="/explain" element={<ProtectedRoute><XAIExplainer /></ProtectedRoute>} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
