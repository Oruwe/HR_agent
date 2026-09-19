import { NavLink, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";
import IntakePage from "./pages/candidate/IntakePage";
import InterviewPage from "./pages/candidate/InterviewPage";
import ResultsPage from "./pages/candidate/ResultsPage";
import DashboardPage from "./pages/admin/DashboardPage";

function CandidateShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">HR Talent Evaluator</span>
          <span className="brand-sub">sub-150ms screening agent</span>
        </div>
        <nav className="nav-links">
          <NavLink to="/" end>
            Candidate call
          </NavLink>
          <NavLink to="/admin">Admin</NavLink>
        </nav>
      </header>
      <main className="app-main">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      {/* The Admin Dashboard is a distinct manager-facing surface with its own
          chrome and theme (see DashboardPage / admin-theme.css) -- it is
          never wrapped in the Candidate App's shell, and it never links back
          into a live interview. */}
      <Route path="/admin" element={<DashboardPage />} />
      <Route path="/admin/sessions/:sessionId" element={<DashboardPage />} />

      <Route path="/" element={<CandidateShell><IntakePage /></CandidateShell>} />
      <Route path="/interview" element={<CandidateShell><InterviewPage /></CandidateShell>} />
      <Route path="/results" element={<CandidateShell><ResultsPage /></CandidateShell>} />
    </Routes>
  );
}
