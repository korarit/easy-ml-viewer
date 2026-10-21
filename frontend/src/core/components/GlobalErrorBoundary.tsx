import { Component, type ReactNode } from 'react';
import { TriangleAlert } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

/** Last-resort crash boundary so one panel never blanks the whole studio. */
export function GlobalErrorBoundary({ children }: Props) {
  return <Boundary>{children}</Boundary>;
}

class Boundary extends Component<Props, State> {
  state: State = { hasError: false, message: '' };

  static getDerivedStateFromError(err: unknown): State {
    return { hasError: true, message: err instanceof Error ? err.message : String(err) };
  }

  componentDidCatch(err: unknown) {
    // eslint-disable-next-line no-console
    console.error('[GlobalErrorBoundary]', err);
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div
        className="flex h-full flex-col items-center justify-center gap-3 p-8"
        style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)' }}
      >
        <TriangleAlert size={28} style={{ color: 'var(--ml-anomaly)' }} />
        <h1 className="text-base font-semibold">Something crashed in this panel</h1>
        <p className="font-mono-tabular max-w-lg text-center text-xs" style={{ color: 'var(--text-secondary)' }}>
          {this.state.message}
        </p>
        <button
          className="rounded-md px-4 py-2 text-sm font-medium"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)' }}
          onClick={() => this.setState({ hasError: false, message: '' })}
        >
          Try again
        </button>
      </div>
    );
  }
}
