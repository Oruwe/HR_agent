import { NavLink, Route, Routes } from "react-router-dom";
import IntakePage from "./pages/candidate/IntakePage";
import InterviewPage from "./pages/candidate/InterviewPage";
import ResultsPage from "./pages/candidate/ResultsPage";
import DashboardPage from "./pages/admin/DashboardPage";
import SessionDetailPage from "./pages/admin/SessionDetailPage";

function TopBar() {
  return (
    <header className="top-bar">
      <div className="brand">
        <span className="brand-mark" />
        <span className="brand-name">HR Talent Evaluator</span>
        <span className="brand-sub">sub-160ms screening agent</span>
      </div>
      <nav className="nav-links">
        <NavLink to="/" end>
          Candidate call
        </NavLink>
        <NavLink to="/admin">Admin</NavLink>
      </nav>
    </header>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <TopBar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<IntakePage />} />
          <Route path="/interview" element={<InterviewPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/admin" element={<DashboardPage />} />
          <Route path="/admin/sessions/:sessionId" element={<SessionDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
