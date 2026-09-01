#!/usr/bin/env bash

set -euo pipefail

DEFAULT_MODEL="deepseek-v4-flash"
DEFAULT_BASE_URL="https://api.deepseek.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${BARE_AGENT_ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ -e "${ENV_FILE}" ]]; then
  printf '.env already exists. Replace it? [y/N]: ' >&2
  IFS= read -r replace
  case "${replace}" in
    y | Y | yes | YES) ;;
    *)
      printf 'Configuration cancelled; existing .env was not changed.\n' >&2
      exit 1
      ;;
  esac
fi

printf 'API_KEY (input hidden): ' >&2
IFS= read -r -s api_key
printf '\n' >&2
if [[ -z "${api_key}" ]]; then
  printf 'API key cannot be empty.\n' >&2
  exit 1
fi

printf 'MODEL [%s]: ' "${DEFAULT_MODEL}" >&2
IFS= read -r model
model="${model:-${DEFAULT_MODEL}}"

printf 'BASE_URL [%s]: ' "${DEFAULT_BASE_URL}" >&2
IFS= read -r base_url
base_url="${base_url:-${DEFAULT_BASE_URL}}"

for value in "${api_key}" "${model}" "${base_url}"; do
  if [[ "${value}" == *"'"* ]]; then
    printf "Values must not contain a single quote.\n" >&2
    exit 1
  fi
done

if [[ "${base_url}" != https://* ]]; then
  printf 'BASE_URL must use HTTPS to protect the API key in transit.\n' >&2
  exit 1
fi

env_dir="$(dirname "${ENV_FILE}")"
if [[ ! -d "${env_dir}" ]]; then
  printf 'The destination directory does not exist.\n' >&2
  exit 1
fi

umask 077
temporary_file="$(mktemp "${env_dir}/.bare-agent-env.XXXXXX")"
cleanup() {
  if [[ -n "${temporary_file:-}" && -e "${temporary_file}" ]]; then
    rm -f -- "${temporary_file}"
  fi
}
trap cleanup EXIT

{
  printf "OPENAI_API_KEY='%s'\n" "${api_key}"
  printf "OPENAI_MODEL='%s'\n" "${model}"
  printf "OPENAI_BASE_URL='%s'\n" "${base_url}"
} >"${temporary_file}"

chmod 600 "${temporary_file}"
mv -f -- "${temporary_file}" "${ENV_FILE}"
temporary_file=""
chmod 600 "${ENV_FILE}"

printf 'Configuration saved to the project-local .env file (permissions: 600).\n'
printf 'Run: uv run bare-agent --workspace ./project "your task"\n'
