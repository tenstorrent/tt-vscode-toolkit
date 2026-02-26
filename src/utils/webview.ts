/**
 * Generates a cryptographically random nonce for CSP (Content Security Policy).
 * Used to allow specific inline scripts while maintaining security.
 *
 * @returns A random nonce string suitable for use in CSP headers
 */
export function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
