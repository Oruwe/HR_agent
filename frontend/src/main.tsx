import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import DashboardPage from "./pages/DashboardPage";
import "./styles/dashboard.css";

// One surface, one route. There is no candidate-facing app and no router:
// the product is the manager's dashboard.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DashboardPage />
  </StrictMode>
);
