# BareAgent

BareAgent is a from-scratch command-line coding agent. It will use an
OpenAI-compatible model interface together with locally implemented file and
command tools to inspect a workspace, edit code, run checks, and iterate on a
programming task.

The project is under active development. No agent framework or remotely hosted
code-execution/file tool is used.

## Security

API credentials must be provided through environment variables. Never place a
real credential in source files, documentation, logs, screenshots, videos, or
Git history.
