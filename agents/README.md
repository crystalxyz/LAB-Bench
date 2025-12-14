# LAB-Bench Agents

This directory contains CLI-based agents for evaluating FigQA tasks in LAB-Bench. All agents follow a Harbor-compatible workflow pattern, making them suitable for both standalone evaluation and integration with Harbor.

## Overview

The agents in this directory share a common architecture:
- **Workspace-based execution**: Each task runs in an isolated workspace directory
- **Harbor-style answer format**: Agents write answers to `answer.txt` in `[ANSWER]X[/ANSWER]` format
- **Figure handling**: Images are saved to the workspace and referenced by path
- **CLI-based**: Each agent wraps a command-line tool rather than using SDKs directly

## Architecture

### Base Class: `BaseCliAgent`

[base_cli_agent.py](base_cli_agent.py:42-234)

All agents inherit from `BaseCliAgent`, which provides:

**Common Functionality:**
- Workspace creation (`{agent_prefix}/run_{uuid}/`)
- Figure saving (preserves original filenames)
- Answer parsing from `answer.txt`
- Strict format validation (`[ANSWER]X[/ANSWER]`)
- Error handling and `UnanswerableError` classification

**Template Methods (subclasses must implement):**
- `workspace_prefix`: Directory name for agent artifacts
- `_build_cli_command()`: Construct the CLI command
- `_run_cli()`: Execute the command and handle errors

**Workflow:**
```python
async def run_task(input: AgentInput) -> str:
    1. Create workspace directory: {workspace_prefix}/run_{uuid}/
    2. Save figure to workspace (preserving filename)
    3. Build prompt with MCQ template
    4. Execute CLI command in workspace
    5. Parse answer from answer.txt
    6. Return answer letter (A, B, C, etc.)
```

### Answer Format

Agents must write answers to `answer.txt` in this exact format:
```
[ANSWER]A[/ANSWER]
```

The parser will:
- ✅ First check `answer.txt` for the answer (Harbor-style)
- ✅ Fallback to parsing stdout if answer.txt doesn't exist
- ❌ Reject answers without proper `[ANSWER][/ANSWER]` tags

### Prompt Template

All agents use the same MCQ instruction template that can be found at `base_cli_agent.py`.

## Available Agents

### 1. Gemini CLI Agent

**File:** [gemini_agent.py](gemini_agent.py:25-97)

Uses Google's `@google/gemini-cli` npm package.

**Setup:**
```bash
npm install -g @google/gemini-cli
export GEMINI_API_KEY=your_key  # or GOOGLE_API_KEY
```

**How it works:**
- **CLI command:** `gemini -y -m {model} {prompt}`
- **Figure handling:** Gemini CLI automatically scans the workspace directory (cwd) for images; figures are referenced by path in the prompt
- **Workspace:** `gemini_cli_artifacts/run_{uuid}/`
- **Error handling:**
  - Token limit errors → `UnanswerableError`
  - API 400 errors → `UnanswerableError`
  - Other errors → `RuntimeError`

**Example command:**
```bash
cd LAB-Bench
python -m agents.gemini_agent
```

### 2. Codex CLI Agent

**File:** [codex_agent.py](codex_agent.py:24-114)

Uses OpenAI's Codex CLI tool.

**Setup:**
```bash
# Install Codex CLI (installation method varies)
export OPENAI_API_KEY=your_key  # or CODEX_API_KEY
```


**How it works:**
- **CLI command:** `codex exec -m {model} -s workspace-write --color never --skip-git-repo-check {prompt}`
- **Figure handling:** Prompt-driven (figures saved to workspace, referenced by path in prompt)
- **Workspace:** `codex_artifacts/run_{uuid}/`
- **Sandbox:** `workspace-write` mode allows writing to workspace
- **Debug logging:** Prints stdout and answer.txt content for inspection
- **Error handling:**
  - Context length errors → `UnanswerableError`
  - API 400 errors → `UnanswerableError`
  - Other errors → `RuntimeError`

**Execution**
```bash
cd LAB-Bench
python -m agents.codex_agent
```

## Error Handling

All agents distinguish between two types of errors:

**UnanswerableError** - Task cannot be answered (doesn't count against agent):
- Token/context length exceeded
- API 400/bad request errors
- Image processing failures
- Malformed inputs

**RuntimeError** - Agent execution failed (counts as failure):
- CLI tool not found
- Authentication errors
- Unexpected exceptions
- Invalid command syntax

## Workspace Organization

Each agent run creates an isolated workspace:

```
{workspace_prefix}/
  run_abc123/
    figure.png              # Original figure (filename preserved)
    answer.txt              # Agent's answer (if written)
    (other agent artifacts)
  run_def456/
    ...
```

**Workspace prefixes:**
- `gemini_cli_artifacts` - Gemini CLI agent
- `codex_artifacts` - Codex CLI agent

**Cleanup:** Workspaces persist after evaluation for debugging. You can manually delete workspace directories if needed.



## Harbor Integration

These agents are designed to mimic Harbor's workflow, making them compatible with the Harbor framework:

**Harbor Adapter Pattern:**
1. Harbor builds Docker container with `/app/instruction.md` and figures
2. Agent runs inside container, reads instruction, processes figures
3. Agent writes answer to `/app/answer.txt` in `[ANSWER]X[/ANSWER]` format
4. Verifier (`test.sh`) checks answer.txt against correct answer
5. Harbor collects trajectory data via `populate_context_post_run()`

**Standalone LAB-Bench Pattern (these agents):**
1. Agent creates workspace directory with unique ID
2. Saves figure to workspace (preserving filename)
3. Builds prompt with figure path reference
4. Runs CLI tool in workspace
5. Parses answer from `answer.txt` (or stdout fallback)
6. Returns answer for evaluation

**Key Differences:**
- Harbor uses `/app/` as workspace; standalone uses `{prefix}/run_{uuid}/`
- Harbor passes instruction via file; standalone passes via CLI argument
- Harbor trajectory collection is agent-specific; standalone logs to task_buffer
- Both enforce same answer format: `[ANSWER]X[/ANSWER]`
