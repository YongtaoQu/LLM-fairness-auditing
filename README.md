# Linguistic Fairness in LLMs through a Mongolian Case Study

Cloze-style bilingual evaluation of LLM accuracy and latency for English vs. traditional Mongolian prompts, using OpenRouter.

## Contents

| Path | Description |
|------|-------------|
| `questions.json` | 20 cloze questions (`question_en`, `question_mn_traditional`) |
| `run_experiment.py` | Experiment runner (stdlib only, Python 3.8+) |
| `results/` | Formal run summaries (manifest + CSV/JSONL) |

### Formal results included

| Directory | Model | Search | Thinking |
|-----------|-------|--------|----------|
| `results/qwen_3.7_plus/` | `qwen/qwen3.7-plus` | Exa web search | Off |
| `results/deepseek_v4.1_flash/` | `deepseek/deepseek-v4.1-flash` | Exa web search | Off |
| `results/gemini_3.8_flash/` | `google/gemini-3.8-flash` | Off | Provider-default (cannot disable) |

Each result folder contains `manifest.json`, `results.jsonl`, and `results.csv` (120 trials: 20 questions × 2 languages × 3 passes).

## Setup

```bash
export OPENROUTER_API_KEY='your_key'
# optional
export OPENROUTER_BASE_HTTP_API_URL='https://openrouter.ai/api/v1'
```


## Run

Dry run:

```bash
python3 run_experiment.py --dry-run
```

Pilot (Q01, one pass):

```bash
python3 run_experiment.py \
  --model qwen/qwen3.7-plus \
  --ids Q01 --passes 1 \
  --max-tokens 2048 --timeout 600 \
  --run-dir results/pilot_qwen
```

Full schedule (120 requests):

```bash
python3 run_experiment.py \
  --model deepseek/deepseek-v4.1-flash \
  --passes 3 \
  --max-tokens 2048 --timeout 600 \
  --run-dir results/deepseek_v4.1_flash
```

Useful flags:

- `--search-engine {exa,native,auto,none}` (default `exa`)
- `--thinking` / `--reasoning-max-tokens`
- `--ids Q01,Q02` to subset questions
- Re-running the **same** `--run-dir` resumes unfinished trials; failed trials are not auto-retried

## Notes

- Prompt policy: original question text only; one new conversation per request.
- Gemini endpoints may reject `reasoning.enabled=false`; omitting the field uses provider defaults.
- Raw streaming logs under `raw/` are omitted from this repository to keep the project submittable.
