# verification2

USC EE599. `uv sync`, then `uv run pre-commit install`.

## Jev evidence selection

Jev (TypeSafe) scores each evidence unit by how strongly it supports or
contradicts the candidate answer, then packs the most useful units into a
token budget. Code lives in `src/verification2/jev/`.

Setup: put `AI_GATEWAY_API_KEY=...` (Vercel AI Gateway) in `.env`.

```
uv run pytest                         # unit tests (no API calls)
uv run python scripts/smoke_test.py   # one live call per passage on a toy example
uv run python scripts/usage.py        # token / cost ledger, --by run|day|model
```
