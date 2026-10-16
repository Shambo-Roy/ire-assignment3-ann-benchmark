# AI usage statement

**Tool used:** Claude Code (Anthropic), desktop app, model Claude Opus 5.5.
(The milestone prompt calls the assistant "Codex", but the tool actually used was Claude Code.)

**How it was used:** The author writes a prompt for each milestone. Claude Code inspects the
workspace, writes code, tests and documentation, runs the tests, and reports back. The author
reviews each milestone before the next one starts. The author writes the experimental hypotheses
(`hypothesis.md`) personally.

## File classification

Definitions:
- **human-written**: written by the author with no AI-generated content.
- **AI-assisted**: written by the author with AI suggestions, or AI-written then substantially edited by the author.
- **AI-generated**: written by the AI; the author reviewed it but has not substantially changed it.

Every file below is classified by what has actually happened so far. **Update a row whenever you
edit a file yourself** (for example AI-generated → AI-assisted).

| File | Classification | Notes |
|---|---|---|
| `src/__init__.py` | AI-generated | Milestone 1 |
| `src/config.py` | AI-generated | Milestone 1 |
| `src/data_loader.py` | AI-generated | Milestone 1 |
| `src/environment_info.py` | AI-generated | Milestone 1 |
| `src/validate_data.py` | AI-generated | Milestone 1 |
| `tests/__init__.py` | AI-generated | Milestone 1 (empty) |
| `tests/conftest.py` | AI-generated | Milestone 1 |
| `tests/test_data_loader.py` | AI-generated | Milestone 1 |
| `tests/test_environment_info.py` | AI-generated | Milestone 1 |
| `run_all.sh` | AI-generated | Milestone 1 (partial) |
| `pytest.ini` | AI-generated | Milestone 1 |
| `requirements.txt` | AI-generated | Milestone 1 |
| `.gitignore` | AI-generated | Milestone 1 |
| `README.md` | AI-generated | Milestone 1 |
| `IMPLEMENTATION_LOG.md` | AI-generated | Milestone 1 |
| `hypothesis.md` | AI-generated (template only) | The author must write the hypothesis text |
| `ai_log/ai_statement.md` | AI-generated | Classifications to be checked by the author |

## Prompt / response record

| Milestone | Prompt | Response | Full transcript |
|---|---|---|---|
| 1 | `ai_log/milestone1/prompt.md` (verbatim) | `ai_log/milestone1/response.md` (final summary, verbatim) | `ai_log/transcripts/` — **the author must copy it in, see below** |

### Completeness and how to preserve the log

Claude Code saves every session automatically as a JSONL file on the author's machine:
`%USERPROFILE%\.claude\projects\C--shambo-roy-iiit-ac-in-Assignment-3\<session-id>.jsonl`.
That file has every prompt, every response and every tool call. `prompt.md` and `response.md`
hold only the main prompt and the final reply, so **they are not a complete log by themselves.**

After each milestone session, the author must:
1. Copy that session's `.jsonl` file into `ai_log/transcripts/`. The Milestone 1 session id is
   `9d5f9e15-4e82-4a66-981e-07501b47b7c0`. Copy it **after** the session ends so it includes the last reply.
2. Save any follow-up questions asked in chat, which the JSONL file already captures.
3. Record any other AI tools used (ChatGPT, Copilot, ...) here. They are not captured automatically.
4. Before submitting, skim the transcripts for personal information you do not want to share,
   such as your email address or local paths. Any redaction must be noted here.
