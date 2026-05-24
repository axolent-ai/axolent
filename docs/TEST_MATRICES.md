# Test Matrices

Parametrized cross-cutting behavior tests for AXOLENT.

## Philosophy

Instead of writing individual tests for each language/command/channel combination,
we use `pytest.mark.parametrize` to systematically test the cartesian product of
dimensions that matter. This finds bugs at intersections that individual tests miss.

## Dimensions

| Dimension | Values | Count |
|-----------|--------|-------|
| Languages | de, en, nl, sv, fr, es, it, pt, pl, tr | 10 |
| Commands (with args) | /remember, /learn, /forget, /explain, /memory, /skills, /skill, /usage | 8 |
| Commands (no args) | /reset, /stop, /help, /start, /settings, /onboarding | 6 |
| Channels | normal, reply, long_message, streaming | 4 |

## Matrices

### 1D Matrices (single dimension)

| File | Dimension | Tests | What it verifies |
|------|-----------|-------|------------------|
| `test_language_matrix.py` | Languages (10) | ~130 | Detection, sticky persistence, resolver, system prompt, i18n |
| `test_command_matrix.py` | Commands (14) | ~42 | Handler registration, i18n keys, no-crash smoke, arg propagation |
| `test_channel_matrix.py` | Channels (4) | ~56 | User ID preservation, Unicode handling, rate-limit uniformity, metadata |

### 2D Matrix

| File | Dimensions | Tests | What it verifies |
|------|------------|-------|------------------|
| `test_lang_x_cmd.py` | Languages x Commands (10 x 8) | ~240 | Sticky language survives command invocation, i18n coverage, language lock |

### Models dimension (intentionally excluded)

Most AXOLENT logic is model-agnostic. Model-specific tests live in
`test_application/test_routing/`. Including models in the broad matrix would
create 960+ tests (3 x 10 x 8 x 4) without meaningful additional coverage,
since the model selection happens downstream of language/command/channel processing.

## When to use which matrix size

| Situation | Recommendation |
|-----------|---------------|
| Testing a property that applies to ALL values in one dimension | 1D matrix |
| Testing interaction between two dimensions (e.g. language affects command response) | 2D matrix |
| Testing a property that only fails at specific triple-intersections | 3D matrix (rare) |
| Testing model-specific behavior | Dedicated test in test_routing/, NOT matrix |

## How to add a new language

1. Add the language code to `LANGUAGES` in `bridge/tests/test_matrices/conftest.py`
2. Add a marker text entry to `LANGUAGE_MARKER_TEXTS` dict (must be >6 words, detectable with confidence > 0)
3. Add a corpus entry to the `language_corpus` fixture
4. Run `pytest tests/test_matrices/ -v` to verify all new tests pass
5. If detection fails: check `domain/language.py` marker words for the new language

## How to add a new command

1. Add the command to `COMMANDS_WITH_ARGS` or `COMMANDS_NO_ARGS` in `conftest.py`
2. Add the handler path to `_COMMAND_HANDLERS` in `test_command_matrix.py`
3. Add relevant i18n keys to `_COMMAND_I18N_KEYS` in `test_command_matrix.py`
4. Run `pytest tests/test_matrices/test_command_matrix.py -v`

## How to add a new channel type

1. Add the channel name to `CHANNELS` in `conftest.py`
2. Add handling logic to `_simulate_channel_message()` in `test_channel_matrix.py`
3. Run `pytest tests/test_matrices/test_channel_matrix.py -v`

## Running matrix tests

```bash
# All matrix tests
pytest tests/test_matrices/ -v

# Only language matrix
pytest tests/test_matrices/test_language_matrix.py -v

# Only the 2D language x command matrix
pytest tests/test_matrices/test_lang_x_cmd.py -v

# Filter by marker
pytest -m matrix -v
```

## Marker

All matrix tests are marked with `@pytest.mark.matrix` for selective execution.
Registered in `pyproject.toml`.
