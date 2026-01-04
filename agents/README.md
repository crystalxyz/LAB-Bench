# LAB-Bench Agents

This directory contains CLI-based agents for evaluating FigQA tasks in LAB-Bench. All agents follow a Harbor-compatible workflow pattern, making them suitable for both standalone evaluation and integration with Harbor.

## Overview

The agents in this directory share a common architecture:
- **Workspace-based execution**: Each task runs in an isolated workspace directory
- **Harbor-style answer format**: Agents write answers to `answer.txt` in `[ANSWER]X[/ANSWER]` format
- **Figure handling**: Images are saved to the workspace and referenced by path

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
- Define the prompt template

## Available Agents

### 1. Gemini CLI Agent

**File:** [gemini_agent.py](gemini_agent.py)

Uses Google's `@google/gemini-cli` npm package.

**Setup:**
```bash
npm install -g @google/gemini-cli
export GEMINI_API_KEY=your_key  # or GOOGLE_API_KEY
```

**Usage:**
```bash
cd LAB-Bench

# Full run, timeout 300s, all 181 tasks
python -m agents.gemini_agent

# Debug run, fail fast
python -m agents.gemini_agent --debug --timeout 120
```

### 2. Codex CLI Agent

**File:** [codex_agent.py](codex_agent.py)

Uses OpenAI's Codex CLI tool.

**Setup:**
```bash
# Install Codex CLI (installation method varies)
export OPENAI_API_KEY=your_key
```

**Usage:**
```bash
cd LAB-Bench

# Full run, timeout 300s, all 181 tasks
python -m agents.codex_agent

# Debug run, fail fast
python -m agents.codex_agent --debug --timeout 120
```

## CLI Options

Both agents support the same CLI flags:

| Flag        | Default | Description                      |
| ----------- | ------- | -------------------------------- |
| `--debug`   | `False` | Run in debug mode (only 8 tasks) |
| `--timeout` | `300`   | Timeout in seconds for each task |

## Accuracy Calculation

After a run completes, `result.json` is automatically saved to the artifacts directory.

You can also manually calculate accuracy for an existing artifacts directory:

```bash
python -m agents.calculate_accuracy codex_artifacts_20250103_143022/
```

This outputs a Harbor-compatible `result.json` with:
- **accuracy**: correct / total
- **precision**: correct / sure (excluding "Insufficient information" answers)
- **coverage**: sure / total

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

Each agent run creates a timestamped workspace with isolated task directories:

```
{agent}_artifacts_{YYYYMMDD_HHMMSS}/
  run_0001/
    figure.png              # Original figure (filename preserved)
    answer.txt              # Agent's answer
    expected_answer.txt     # Correct answer (saved after task completes)
    unsure_answer.txt       # "Insufficient info" option letter
    prompt.txt              # Full prompt sent to agent
    {agent}_trajectory.*    # Raw CLI output
  run_0002/
    ...
  result.json               # Auto-generated accuracy report
```

**Workspace prefixes:**
- `gemini_artifacts_{timestamp}` - Gemini CLI agent
- `codex_artifacts_{timestamp}` - Codex CLI agent

**Cleanup:** Workspaces persist after evaluation for debugging. Delete manually if needed.
