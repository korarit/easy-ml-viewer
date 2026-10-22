import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './core/theme/studio.css';

const root = document.getElementById('root');
if (!root) throw new Error('#root element missing');
// Restore persisted theme (default dark). Header toggle writes `emv-theme`.
try {
  const saved = localStorage.getItem('emv-theme');
  document.documentElement.dataset.theme = saved === 'light' ? 'light' : 'dark';
} catch {
  if (!document.documentElement.dataset.theme) document.documentElement.dataset.theme = 'dark';
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
