import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Short name of the area, shown in the message. */
  area?: string;
  /** Rendered instead of the default message (e.g. a 2D fallback for the 3D view). */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

/** Keeps a failure in one view (a lazy chunk that 404s after a redeploy, a WebGL error) from blanking the app. */
export default class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[model-surgery] ${this.props.area ?? "view"} crashed:`, error, info.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    if (this.props.fallback) return this.props.fallback(error, this.reset);
    const chunk = /dynamically imported module|Loading chunk|Failed to fetch/i.test(error.message);
    return (
      <div className="callout bad" role="alert">
        <strong>{this.props.area ?? "This view"} failed to render.</strong>{" "}
        {chunk ? "Part of the app could not be loaded (the site may have been updated). Reload the page." : error.message}
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn sm" onClick={this.reset}>
            Try again
          </button>
          <button className="btn sm ghost" onClick={() => window.location.reload()}>
            Reload page
          </button>
        </div>
      </div>
    );
  }
}
