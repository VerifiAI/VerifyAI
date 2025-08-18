# Firebase Authentication Module

## Overview
This module provides a complete Firebase authentication implementation for React applications. It includes user registration, login, password reset, and route protection components.

## Features
- Email/Password Authentication
- Social Media Authentication (Google, Twitter, GitHub)
- Protected Routes
- Authentication State Management with Redux
- Form Components with Validation

## Directory Structure
```
Authentication/
├── components/         # UI Components
├── config/             # Firebase Configuration
├── containers/         # Page Containers
│   └── Session/        # Authentication Session Management
└── redux/              # Redux State Management
    └── modules/        # Redux Modules
```

## Setup Instructions

### 1. Firebase Configuration
Update the Firebase configuration in `config/config.js` with your Firebase project details:

```javascript
const firebaseConfig = {
  apiKey: 'YOUR_API_KEY',
  authDomain: 'YOUR_PROJECT_ID.firebaseapp.com',
  databaseURL: 'https://YOUR_PROJECT_ID.firebaseio.com',
  projectId: 'YOUR_PROJECT_ID',
  storageBucket: 'YOUR_PROJECT_ID.appspot.com',
  messagingSenderId: 'YOUR_MESSAGING_SENDER_ID'
};
```

### 2. Redux Integration
Import and use the authentication reducer in your Redux store:

```javascript
import { configureStore } from '@reduxjs/toolkit';
import authReducer from './modules/auth';

const store = configureStore({
  reducer: {
    auth: authReducer,
    // other reducers
  }
});
```

### 3. Protected Routes
Use the `ProtectedPage` component to protect routes that require authentication:

```jsx
import ProtectedPage from './containers/Session/ProtectedPage';

// In your routes configuration
<Route 
  path="/protected-route" 
  element={
    <ProtectedPage>
      <YourProtectedComponent />
    </ProtectedPage>
  } 
/>
```

## Authentication Components

### Login
Provides email/password login and social media authentication options.

### Register
Allows users to create a new account with email/password or social media providers.

### Reset Password
Enables users to reset their password via email.

## Redux State Management

The authentication state is managed through Redux with the following properties:

- `loading`: Indicates authentication operations in progress
- `loggedIn`: Boolean indicating if a user is authenticated
- `user`: User profile information
- `uid`: User ID
- `message`: Authentication operation messages (success/error)

## Firebase Rules

Basic Firebase database rules are included in `firebase.rules.json`. Customize these rules based on your application's security requirements.