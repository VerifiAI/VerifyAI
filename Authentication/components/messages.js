/*
 * Form Messages
 *
 * This contains all the text for the Form components.
 */
import { defineMessages } from 'react-intl';

export const scope = 'boilerplate.components.Forms';

export default defineMessages({
  login: {
    id: `${scope}.login`,
    defaultMessage: 'Login',
  },
  loginOr: {
    id: `${scope}.login_or`,
    defaultMessage: 'Or sign in with',
  },
  loginFieldEmail: {
    id: `${scope}.login_field_email`,
    defaultMessage: 'Your Email',
  },
  loginFieldPassword: {
    id: `${scope}.login_field_password`,
    defaultMessage: 'Your Password',
  },
  loginForgotPassword: {
    id: `${scope}.login_forgot_password`,
    defaultMessage: 'Forgot Password',
  },
  loginRemember: {
    id: `${scope}.login_remember`,
    defaultMessage: 'Remember',
  },
  loginButtonContinue: {
    id: `${scope}.login_button_continue`,
    defaultMessage: 'Continue',
  },
  toRegister: {
    id: `${scope}.to_register`,
    defaultMessage: 'Don\'t have an account ?',
  },
  register: {
    id: `${scope}.register`,
    defaultMessage: 'Register',
  },
  registerOr: {
    id: `${scope}.register_or`,
    defaultMessage: 'Or register with',
  },
  registerButtonContinue: {
    id: `${scope}.register_button_continue`,
    defaultMessage: 'Continue',
  },
  registerFieldName: {
    id: `${scope}.register_field_name`,
    defaultMessage: 'Your Name',
  },
  registerFieldEmail: {
    id: `${scope}.register_field_email`,
    defaultMessage: 'Your Email',
  },
  registerFieldPassword: {
    id: `${scope}.register_field_password`,
    defaultMessage: 'Your Password',
  },
  registerFieldRetypePassword: {
    id: `${scope}.register_field_retype_password`,
    defaultMessage: 'Re-type Password',
  },
  registerAcceptTerms: {
    id: `${scope}.register_accept_terms`,
    defaultMessage: 'I have read and accept the terms of service.',
  },
  toLogin: {
    id: `${scope}.to_login`,
    defaultMessage: 'Already have account ?',
  },
  reset: {
    id: `${scope}.reset`,
    defaultMessage: 'Reset Password',
  },
  resetFieldEmail: {
    id: `${scope}.reset_field_email`,
    defaultMessage: 'Your Email',
  },
  resetFieldPassword: {
    id: `${scope}.reset_field_password`,
    defaultMessage: 'Your New Password',
  },
  resetFieldRetypePassword: {
    id: `${scope}.reset_field_retype_password`,
    defaultMessage: 'Re-type Password',
  },
  resetButtonContinue: {
    id: `${scope}.reset_button_continue`,
    defaultMessage: 'Continue',
  },
  lockHint: {
    id: `${scope}.lock_hint`,
    defaultMessage: 'Enter your password to continue',
  },
  requiredForm: {
    id: `${scope}.required_form`,
    defaultMessage: 'This field is required',
  },
  emailRequired: {
    id: `${scope}.email_required`,
    defaultMessage: 'Email is required',
  },
  passwordRequired: {
    id: `${scope}.password_required`,
    defaultMessage: 'Password is required',
  },
  passwordsNotMatch: {
    id: `${scope}.passwords_not_match`,
    defaultMessage: 'Passwords do not match',
  },
  termRequired: {
    id: `${scope}.term_required`,
    defaultMessage: 'Please read and check the Terms and Condition',
  },
  checkboxRequired: {
    id: `${scope}.checkbox_required`,
    defaultMessage: 'This checkbox is required',
  },
  radioRequired: {
    id: `${scope}.radio_required`,
    defaultMessage: 'Please choose one',
  },
  minLength: {
    id: `${scope}.min_length`,
    defaultMessage: 'Minimal {length} characters',
  },
  emailInvalid: {
    id: `${scope}.email_invalid`,
    defaultMessage: 'Invalid email address',
  },
  passwordInvalid: {
    id: `${scope}.password_invalid`,
    defaultMessage: 'Password must be at least 8 characters and contain number, uppercase, and lowercase character',
  },
  checkout: {
    id: `${scope}.checkout`,
    defaultMessage: 'Checkout',
  },
  checkoutBilling: {
    id: `${scope}.checkout_billing`,
    defaultMessage: 'Billing Address',
  },
  checkoutBillingDesc: {
    id: `${scope}.checkout_billing_desc`,
    defaultMessage: 'Where should we send your receipt?',
  },
  checkoutShipping: {
    id: `${scope}.checkout_shipping`,
    defaultMessage: 'Shipping Address',
  },
  checkoutShippingDesc: {
    id: `${scope}.checkout_shipping_desc`,
    defaultMessage: 'Where should we send your stuff?',
  },
  checkoutPayment: {
    id: `${scope}.checkout_payment`,
    defaultMessage: 'Payment',
  },
  checkoutPaymentDesc: {
    id: `${scope}.checkout_payment_desc`,
    defaultMessage: 'How would you like to pay?',
  },
  checkoutButtonOrder: {
    id: `${scope}.checkout_button_order`,
    defaultMessage: 'Place Order',
  },
  checkoutButtonPrev: {
    id: `${scope}.checkout_button_prev`,
    defaultMessage: 'Previous Step',
  },
  checkoutButtonNext: {
    id: `${scope}.checkout_button_next`,
    defaultMessage: 'Next Step',
  },
  checkoutLabelCardNumber: {
    id: `${scope}.checkout_label_card_number`,
    defaultMessage: 'Card Number',
  },
  checkoutLabelExpired: {
    id: `${scope}.checkout_label_expired`,
    defaultMessage: 'Expired',
  },
  checkoutLabelCVV: {
    id: `${scope}.checkout_label_cvv`,
    defaultMessage: 'CVV / CVC',
  },
  checkoutLabelName: {
    id: `${scope}.checkout_label_name`,
    defaultMessage: 'Name on Card',
  },
  checkoutLabelPhone: {
    id: `${scope}.checkout_label_phone`,
    defaultMessage: 'Phone Number',
  },
  checkoutLabelCompany: {
    id: `${scope}.checkout_label_company`,
    defaultMessage: 'Company',
  },
  checkoutLabelFirstName: {
    id: `${scope}.checkout_label_firstname`,
    defaultMessage: 'First Name',
  },
  checkoutLabelLastName: {
    id: `${scope}.checkout_label_lastname`,
    defaultMessage: 'Last Name',
  },
  checkoutLabelAddress1: {
    id: `${scope}.checkout_label_address1`,
    defaultMessage: 'Address line 1',
  },
  checkoutLabelAddress2: {
    id: `${scope}.checkout_label_address2`,
    defaultMessage: 'Address line 2',
  },
  checkoutLabelCity: {
    id: `${scope}.checkout_label_city`,
    defaultMessage: 'City',
  },
  checkoutLabelState: {
    id: `${scope}.checkout_label_state`,
    defaultMessage: 'State/Province/Region',
  },
  checkoutLabelPostalCode: {
    id: `${scope}.checkout_label_postal`,
    defaultMessage: 'Zip / Postal Code',
  },
  checkoutLabelCountry: {
    id: `${scope}.checkout_label_country`,
    defaultMessage: 'Country',
  },
  checkoutLabelUseAddress: {
    id: `${scope}.checkout_label_use_address`,
    defaultMessage: 'Use this address for payment details',
  },
  checkoutError: {
    id: `${scope}.checkout_error`,
    defaultMessage: 'Please fix the errors',
  },
  checkoutErrorName: {
    id: `${scope}.checkout_error_name`,
    defaultMessage: 'Name on card is required',
  },
  checkoutErrorCardNumber: {
    id: `${scope}.checkout_error_card_number`,
    defaultMessage: 'Wrong card number',
  },
  checkoutErrorExpired: {
    id: `${scope}.checkout_error_expired`,
    defaultMessage: 'Wrong expiration date',
  },
  checkoutErrorCVV: {
    id: `${scope}.checkout_error_cvv`,
    defaultMessage: 'Wrong CVV / CVC format',
  },
  checkoutErrorPhone: {
    id: `${scope}.checkout_error_phone`,
    defaultMessage: 'Phone number is required',
  },
  checkoutErrorCompany: {
    id: `${scope}.checkout_error_company`,
    defaultMessage: 'Company name is required',
  },
  checkoutErrorFirstName: {
    id: `${scope}.checkout_error_firstname`,
    defaultMessage: 'First name is required',
  },
  checkoutErrorLastName: {
    id: `${scope}.checkout_error_lastname`,
    defaultMessage: 'Last name is required',
  },
  checkoutErrorAddress1: {
    id: `${scope}.checkout_error_address1`,
    defaultMessage: 'Address line 1 is required',
  },
  checkoutErrorCity: {
    id: `${scope}.checkout_error_city`,
    defaultMessage: 'City is required',
  },
  checkoutErrorPostalCode: {
    id: `${scope}.checkout_error_postal`,
    defaultMessage: 'Zip / Postal code is required',
  },
  checkoutErrorCountry: {
    id: `${scope}.checkout_error_country`,
    defaultMessage: 'Country is required',
  },
  contactName: {
    id: `${scope}.contact_name`,
    defaultMessage: 'Name',
  },
  contactEmail: {
    id: `${scope}.contact_email`,
    defaultMessage: 'Email',
  },
  contactPhone: {
    id: `${scope}.contact_phone`,
    defaultMessage: 'Phone',
  },
  contactCompany: {
    id: `${scope}.contact_company`,
    defaultMessage: 'Company',
  },
  contactAddress: {
    id: `${scope}.contact_address`,
    defaultMessage: 'Address',
  },
  contactMessage: {
    id: `${scope}.contact_message`,
    defaultMessage: 'Your Message',
  },
  contactButton: {
    id: `${scope}.contact_button`,
    defaultMessage: 'Send',
  },
  loginFailed: {
    id: `${scope}.login_failed`,
    defaultMessage: 'Login Failed',
  },
  loginSuccess: {
    id: `${scope}.login_success`,
    defaultMessage: 'Login Success',
  },
  registrationFailed: {
    id: `${scope}.registration_failed`,
    defaultMessage: 'Registration Failed',
  },
  registrationSuccess: {
    id: `${scope}.registration_success`,
    defaultMessage: 'Registration Success',
  },
  passwordResetFailed: {
    id: `${scope}.password_reset_failed`,
    defaultMessage: 'Password Reset Failed',
  },
  passwordResetSuccess: {
    id: `${scope}.password_reset_success`,
    defaultMessage: 'Password Reset Success',
  },
});