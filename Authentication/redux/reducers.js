/**
 * Combine all reducers in this file and export the combined reducers.
 */
import { combineSlices } from '@reduxjs/toolkit';
import language from 'containers/LanguageProvider/reducer';
import authReducer from './modules/auth';

/**
 * Creates the main reducer with the dynamically injected ones
 */

export default combineSlices({
  auth: authReducer,
  language
});