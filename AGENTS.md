# Repository Guidelines

never edit the main readme. this is hand writtin and should remain that way

## Coding Style & Naming Conventions
  1 - Match the existing 4-space indentation module-level functions
  2 - camelCase naming used throughout. restricted to two words when possible in camelCase
  3 - Keep CLI argument parsing in `main.py` slim and push logic into reusable helpers. 
  4 - Type hints are sprinkled where clarity helps; add them for new public helpers.
  5 - Prefer explicit dictionary keys over dynamic attribute access for profile fields, and 
  6 - document non-obvious calculations with a single-line comment. 
  7 - attempt to keep line lengths to ≤80 characters for readability in terminal windows. 
  8 - Where possible, avoid for loops in favor of vectorized Pandas/Numpy operations for performance.
  9 - use section spacers using lines of `#` to visually separate different sections of code, such as imports, function definitions, and main execution blocks. This helps improve readability and organization of the code.
  10 - structure files top down to follow the flow of execution: imports → helper functions → main execution. avoid jumping around the file to find related code.
  11 - never define variables within loops or conditionals. define them at the top of the function or module to keep variable scope clear and avoid confusion about where variables are defined and used.

## Simplicity Rules                                                                                                                        
  1 - Avoid over-engineering; if a simple solution works, use it. 
       Don’t add complexity for features that aren’t currently needed or for hypothetical future use cases.
  2 - Avoid optional parameters and toggles; if a feature is needed, add it directly. 
       If it’s not needed, don’t add it. This keeps the codebase lean and focused on the core functionality without unnecessary complexity.
  3 - Assume inputs are valid; do not add runtime guards or existence checks.
  4 - Prefer straight‑line code without try/except; let failures surface. 
      keep guardrails for math (devision of 0), edge case indexing bounds and file io.
  5 - Keep modules focused; small, composable helpers over heavy wrappers.
  6 - Use atomic `os.replace` for writes (best‑effort; no retries or locks).
  7 - Keep CLI and bash runners minimal; remove optional toggles and branches.
  8 - no wrappers around existing code for new functionality. write new code instead.
  9 - no fall back parameters. let errors for missing params surface.
  10 - for loops and if/else blocks use simle letter variables. eg "for i in range(x)" etc. avoid descriptive variable names in loops and conditionals to keep them concise and focused on the logic rather than the semantics of the variables.

## Interaction Rules
  1 - never jump straight to coding, always confirm changes first.
  2 - for repo context, check `docs/README.md` first and search `docs/`
      before scanning the wider codebase.

## Roadmap

### Engine boundary (keep it small)
- Core engine scope is strictly:
  - rolling-window signals and math
  - flag generation + context/state machine for inserting signals
- Keep out of the core engine:
  - file I/O, cache, JSON parsing, plotting, Binance/networking
- Prefer SoA (struct-of-arrays) layouts and aligned arrays so the eventual C
  port can run N configs in parallel (tuner sweeps).
- Never introduce non-causal behavior:
  - no centered gradients (e.g. center-referenced `np.gradient`)
  - no future leakage in rolling stats (use past-only mean/std, running peaks)
  - macro→micro alignment must use last-known sample (no interpolation)

### Phases
1) Python R&D (this repo)
  - iterate on math/flags with cached klines only
  - keep CLI/scripts and JSON config behavior stable
  - keep `engine_core.py` cache-free and deterministic (cache stays external)
  - add charts/metrics that help validate causal behavior
  - maintain a deterministic "golden" reference output to verify later ports

2) C core engine + fast sweeps
  - port only: context, signal generation, flags, and signal insertion logic
  - implement SoA sweeps (each config is a column of aligned parameter arrays)
  - keep caching and plotting out of C (Python remains orchestration layer)

3) explore ML/AI options.
