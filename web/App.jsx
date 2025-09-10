import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { auth } from './firebase.js';
import { onAuthStateChanged, getRedirectResult } from 'firebase/auth';
import Login from './components/Login.jsx';
import Dashboard from './components/Dashboard.jsx';
import ProtectedRoute from './components/ProtectedRoute.jsx';
import './style.css';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Handle redirect result from Google authentication
    getRedirectResult(auth)
      .then((result) => {
        if (result) {
          console.log('Redirect result:', result);
          // User is signed in after redirect
          setUser(result.user);
        }
        // Continue with normal auth state monitoring
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
          console.log('Auth state changed:', currentUser);
          setUser(currentUser);
          setLoading(false);
        });
        return () => unsubscribe();
      })
      .catch((error) => {
        console.error('Redirect error in App:', error);
        setLoading(false);
      });
  }, []);

  return (
    <Router>
      <div className="App">
        <Routes>
          {/* Default route - redirect based on authentication status */}
          <Route 
            path="/" 
            element={
              loading ? (
                <div style={{
                  minHeight: '100vh',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <p>Loading...</p>
                </div>
              ) : (
                user ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />
              )
            } 
          />
          
          {/* Login route */}
          <Route 
            path="/login" 
            element={
              user ? <Navigate to="/dashboard" replace /> : <Login />
            } 
          />
          
          {/* Protected Dashboard route */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute user={user} loading={loading}>
                <Dashboard user={user} />
              </ProtectedRoute>
            } 
          />
          
          {/* Catch all route - redirect to home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;