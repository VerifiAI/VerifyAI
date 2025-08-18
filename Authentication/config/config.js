const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || 'AIzaSyARfUbiC82TQsHKTi-hfKEPLVR790maexA',
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || 'verifyai-1053a.firebaseapp.com',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || 'verifyai-1053a',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || 'verifyai-1053a.firebasestorage.app',
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '976752504643',
  appId: import.meta.env.VITE_FIREBASE_APP_ID || '1:976752504643:web:043116c43af77b186bc9f6',
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID || 'G-0SBXNNFGGB'
};

export default firebaseConfig;