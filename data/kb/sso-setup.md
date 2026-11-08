# Setting up SSO (Google Workspace and Okta)

SSO is available on the Pro and Enterprise plans.

1. Go to Settings → Security → Single sign-on.
2. Choose Google Workspace or Okta (SAML 2.0).
3. Paste the IdP metadata URL and click Verify.
4. Turn on "Require SSO" only after one admin has signed in successfully.

Common error: "SAML response signature invalid" means the certificate in the IdP was rotated.
Re-download the metadata and paste it again.
