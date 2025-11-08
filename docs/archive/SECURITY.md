# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainer directly at: hunter@shannonlabs.dev
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity (critical issues within 30 days)

### Disclosure Policy

- We follow coordinated disclosure practices
- Security advisories will be published after fixes are released
- Credit will be given to reporters unless anonymity is requested

## Security Best Practices for Users

When using this software:

1. **Keep dependencies updated**: Run `pip install --upgrade -r requirements.txt` regularly
2. **Validate inputs**: Always sanitize user inputs when running scripts
3. **Secure data**: Protect sensitive simulation data and configuration files
4. **Report issues**: If you notice unusual behavior, report it

## Known Security Considerations

This is a scientific simulation framework intended for research environments:

- **Not designed for untrusted inputs**: Validate all configuration files
- **File system access**: Scripts read/write files in working directories
- **No authentication layer**: Not designed for multi-user or web-facing deployments
- **Resource usage**: Some simulations may consume significant compute resources

## Third-Party Dependencies

We rely on established scientific Python packages (NumPy, SciPy, etc.). Monitor:
- [NumPy Security Advisories](https://github.com/numpy/numpy/security/advisories)
- [PyPI Advisory Database](https://github.com/pypa/advisory-database)

For dependency vulnerabilities, we aim to update within 30 days of disclosure.
