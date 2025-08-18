# Firebase Authentication Integration

This project now includes Firebase Authentication with the following setup:

## Files Added/Modified

### Configuration Files
- `.env` - Contains Firebase configuration variables
- `firebase.js` - Firebase initialization and exports
- `config/config.js` - Updated Firebase configuration using environment variables

### Authentication Components
- `AuthDemo.js` - Simple React authentication demo component
- `App.jsx` - Main React application component
- `main-react.jsx` - React entry point
- `auth.html` - Dedicated page for authentication demo

### Authentication System Files (from Authentication.zip)
- `components/` - Login, Register, Reset forms and related components
- `containers/` - Authentication containers and session management
- `redux/` - Redux store configuration and authentication modules

## Firebase Configuration

The following Firebase configuration is set up in your `.env` file:

```
VITE_FIREBASE_API_KEY=AIzaSyARfUbiC82TQsHKTi-hfKEPLVR790maexA
VITE_FIREBASE_AUTH_DOMAIN=verifyai-1053a.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=verifyai-1053a
VITE_FIREBASE_STORAGE_BUCKET=verifyai-1053a.firebasestorage.app
VITE_FIREBASE_MESSAGING_SENDER_ID=976752504643
VITE_FIREBASE_APP_ID=1:976752504643:web:043116c43af77b186bc9f6
VITE_FIREBASE_MEASUREMENT_ID=G-0SBXNNFGGB
```

## Dependencies Installed

The following packages have been installed:

- `firebase` - Firebase SDK
- `react` & `react-dom` - React framework
- `redux`, `react-redux`, `redux-thunk` - State management
- `@mui/material`, `@mui/icons-material` - Material-UI components
- `@emotion/react`, `@emotion/styled` - Styling
- `react-router-dom` - Routing
- `react-intl` - Internationalization
- `formik`, `yup` - Form handling and validation
- `prop-types` - Type checking

## How to Use

1. **Access the Authentication Demo**: 
   - Visit `http://localhost:3000/auth.html` to see the authentication demo
   - Or click "Try Authentication" button on the main site

2. **Test Authentication**:
   - Register a new account with email and password
   - Login with existing credentials
   - Logout functionality

3. **Integration with Existing Project**:
   - The main landing page (`index.html`) remains unchanged
   - Authentication components are available in the `components/` folder
   - Redux store is configured in `redux/` folder
   - Use the components from `containers/` for full-featured authentication pages

## Development

To run the project:
```bash
npm run dev
```

The authentication demo will be available at:
- Main site: `http://localhost:3000/`
- Auth demo: `http://localhost:3000/auth.html`

## Security Notes

- Environment variables are used for Firebase configuration
- Never commit sensitive API keys to version control
- The current configuration is for development purposes
- For production, ensure proper Firebase security rules are configured

## Next Steps

1. Integrate authentication components into your main application
2. Set up Firebase security rules
3. Configure user roles and permissions
4. Add protected routes using the `ProtectedPage` component
5. Customize the UI to match your design system