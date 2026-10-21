import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './core/theme/studio.css';

const root = document.getElementById('root');
if (!root) throw new Error('#root element missing');
if (!document.documentElement.dataset.theme) document.documentElement.dataset.theme = 'dark';

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
