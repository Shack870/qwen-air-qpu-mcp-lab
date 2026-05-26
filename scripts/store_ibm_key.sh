#!/usr/bin/env bash
set -euo pipefail

echo "Paste your IBM Quantum API key. It will be stored in macOS Keychain."
read -r -s -p "IBM Quantum API key: " IBM_KEY
echo
security add-generic-password -a "$USER" -s ibm_quantum_api_key -w "$IBM_KEY" -U

echo "Optional: paste your IBM Quantum instance/CRN/name, or press Enter to skip."
read -r -p "IBM Quantum instance: " IBM_INSTANCE
if [[ -n "$IBM_INSTANCE" ]]; then
  security add-generic-password -a "$USER" -s ibm_quantum_instance_crn -w "$IBM_INSTANCE" -U
fi

echo "Stored. The harness can detect the credential, but it will not print it."

