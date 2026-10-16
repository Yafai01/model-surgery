import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import { applyTheme, loadPrefs } from "./prefs";
import "./styles.css";

applyTheme(loadPrefs().theme); // before the first paint, so there is no flash of the wrong theme

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary area="Model Surgery">
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
