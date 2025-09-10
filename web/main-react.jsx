import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import AuthDemo from './AuthDemo.jsx';

// Create root and render the React app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);