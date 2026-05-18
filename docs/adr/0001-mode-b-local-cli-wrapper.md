# ADR-0001: Mode B Local CLI Wrapper

**Status:** Accepted
**Date:** 2026-03-01
**Decision makers:** Jessica (AXOLENT AI)

## Context

AXOLENT AI needs to interface with Claude (Anthropic's LLM). There are two
architectural approaches:

**Mode A (rejected):** Intercept or proxy the user's OAuth token to make
direct API calls. This provides maximum control over the request/response
cycle but violates Anthropic's Terms of Service and creates security risks
(token storage, token leakage, credential management).

**Mode B (chosen):** Spawn the official `claude` CLI as a local subprocess.
The user has their own Claude Pro/Max subscription installed on their machine.
AXOLENT AI sends prompts via stdin and reads responses from stdout/stderr.

Anthropic explicitly permits local CLI tool usage with the user's own
subscription. There is no ambiguity about Terms of Service compliance.

## Decision

Use Mode B: local CLI wrapper that spawns `claude` as a subprocess.

**Implementation:**

* `infrastructure/claude_process_pool.py` manages persistent subprocesses
  keyed by `(user_id, chat_id)`.
* Prompts are sent via stdin pipe.
* Responses are parsed as NDJSON stream events.
* Subprocess TTL is configurable (default: 60 minutes).
* No Anthropic SDK is imported anywhere in the codebase.
* No API key or OAuth token is read, stored, or proxied.

## Consequences

### Positive

* **TOS compliance:** No risk of account suspension or legal issues.
* **No token management:** No need to handle, store, or rotate API keys.
* **No cloud proxy:** No server between user and Anthropic. Eliminates an
  entire attack surface.
* **User controls billing:** The user manages their own subscription and
  usage limits directly with Anthropic.
* **Privacy:** Anthropic sees the user's normal CLI usage, nothing more.

### Negative

* **Dependency on CLI availability:** If the `claude` CLI is not installed
  or not logged in, the bot cannot function.
* **Subprocess overhead:** Process creation and management adds latency
  compared to direct API calls. Mitigated by persistent process pool
  (74% faster than cold-start subprocess).
* **Limited control:** Cannot use Anthropic SDK features directly (e.g.,
  structured output, tool use via API). Must work within CLI capabilities.
* **User prerequisite:** Every user needs their own Pro/Max subscription.
  There is no free tier or trial.

### Constraints

* No production code may import the Anthropic Python SDK.
* No code may read from `~/.claude` or any token storage location.
* No HTTP requests may be made directly to Anthropic API endpoints.
* All provider interactions must go through the subprocess wrapper.
* Code reviews must verify Mode B compliance.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Mode A (OAuth proxy) | Violates Anthropic TOS, creates security liability |
| Direct API with user-provided key | Requires key management, leakage risk |
| Cloud-hosted proxy | Adds server infrastructure, privacy concerns |
| Anthropic SDK integration | Bypasses CLI, unclear TOS status for wrapper apps |

## References

* [README.md](../../README.md): Mode B explanation
* [SECURITY.md](../../SECURITY.md): Security architecture
* [docs/THREAT_MODEL.md](../THREAT_MODEL.md): Trust boundary and adversary model
