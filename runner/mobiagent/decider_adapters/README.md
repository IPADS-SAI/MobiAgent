# Decider Adapter Contract

This directory contains protocol-specific adapters that bridge model outputs into
the shared MobiAgent runtime.

Each new model integration should add one adapter module and expose one
`DeciderAdapter` via `get_adapter()`. The adapter must implement exactly these
three protocol-specific interfaces:

1. `build_messages(task, history, screenshot_b64, use_e2e, device_type)`
   Build the prompt/messages for that model.
2. `parse_response(response_str)`
   Parse the raw model output into the shared decider response shape:
   `{"reasoning": str, "action": str, "parameters": dict}`.
3. `validate_response(response_dict, use_e2e)`
   Enforce protocol-specific rules that are not part of the shared runtime
   schema.

The shared runtime schema is intentionally not defined in the adapters. After an
adapter parses a response, `runner/mobiagent/mobiagent.py` runs
`validate_action_parameters(...)` to normalize the response into the canonical
runtime schema consumed by the execution handlers.

Rule of thumb:

- Put model prompt format, output parsing, and protocol field checks here.
- Put cross-model action execution schema in `mobiagent.py`.
- If a new model needs extra compatibility logic, convert it here and return the
  shared action names already used by the runtime.
