# Security

## Secrets

Do not commit IBM Quantum API keys, instance identifiers, model-access tokens, or
private job/account metadata.

Preferred local storage on macOS:

```bash
./scripts/store_ibm_key.sh
```

The harness reads:

- `IBM_QUANTUM_API_KEY`, or macOS Keychain service `ibm_quantum_api_key`
- `IBM_QUANTUM_INSTANCE`, or macOS Keychain service `ibm_quantum_instance_crn`

The MCP server exposes credential status only as source labels such as
`environment` or `macOS keychain`; it should never return secret values.

## QPU Jobs

Real IBM Quantum jobs require an explicit `--allow-real-qpu` flag or equivalent
tool argument. Keep jobs small, inspect the backend, and treat Open Plan time as a
limited shared resource.

## Local Process Control

Some clean-room scripts temporarily pause nonessential user processes to reduce
benchmark contention. They should not kill critical macOS services. If you adapt
those scripts, keep a `finally`/trap path that resumes anything paused.

## Reporting Issues

If you find a secret in the repository history, revoke the credential first, then
open a private issue or contact the maintainer before public disclosure.
