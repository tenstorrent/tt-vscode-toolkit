# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The Tenstorrent team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
* **tsingletary@tenstorrent.com**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include in Your Report

To help us better understand the nature and scope of the possible issue, please include as much of the following information as possible:

* Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

### Disclosure Policy

* Security issues should be reported privately first
* We will investigate all legitimate reports and do our best to quickly address the issue
* We will acknowledge your contribution in release notes (unless you prefer to remain anonymous)
* We follow a coordinated disclosure model and will notify you before publicly disclosing the vulnerability

## Security Best Practices for Users

When using the TT-VSCode-Toolkit:

1. **Keep your extension updated** - Always use the latest version to ensure you have the latest security patches
2. **Review lesson content** - Be cautious when executing commands from lessons, especially those that modify your system
3. **Use trusted sources** - Only add lessons from trusted sources to your workspace
4. **Check permissions** - Review what permissions the extension requests during installation
5. **Report suspicious behavior** - If you notice any unexpected behavior, report it immediately

## Security Features

This extension includes the following security features:

* **Command validation** - Terminal commands are validated before execution
* **Sandboxed execution** - Commands run in isolated terminal instances
* **Content security** - Webviews use Content Security Policy to prevent XSS attacks
* **Safe markdown rendering** - Markdown content is sanitized before rendering

## Vulnerability Disclosure Timeline

1. **Day 0**: Security vulnerability reported to tsingletary@tenstorrent.com
2. **Day 1-2**: Initial response acknowledging receipt
3. **Day 3-7**: Investigation and assessment of the vulnerability
4. **Day 8-14**: Development of a fix (timeline may vary based on severity)
5. **Day 15-21**: Release of patched version
6. **Day 22+**: Public disclosure (coordinated with reporter)

## Security Updates

Security updates will be released as needed and will be clearly marked in the changelog. Critical security updates may be released outside of the regular release schedule.

## Third-Party Dependencies

This extension relies on several third-party dependencies. We regularly:
* Monitor security advisories for our dependencies
* Update dependencies to address known vulnerabilities
* Use automated tools (Dependabot, npm audit) to identify potential issues

## Contact

For any security-related questions or concerns, please contact:
* Email: tsingletary@tenstorrent.com
* GitHub: [TT-VSCode-Toolkit Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues) (for non-security bugs only)

---

Thank you for helping keep the TT-VSCode-Toolkit and our users safe!
