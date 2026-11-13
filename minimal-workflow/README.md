Note: All text below (and code in the folder) was written by codex.

### Toy implementation

The `minimal-workflow/` folder runs this workflow against models on OpenRouter.
It uses Python's standard library plus the repository's `support_eval.py`; no
extra packages are required. The runner makes three separate calls per
model/question: answer the question, report certainty, then list `K` similar
situations. It scores the cited situations with the external TF-IDF cosine
baseline described above. This measures textual similarity, not whether an
example is a real or valid source.

Before running, edit `MODELS` in `minimal-workflow/main.py` to contain the
OpenRouter model IDs to compare. Set `OPENROUTER_API_KEY` in that file, or
export it in the shell. `K` is also a global in `main.py` and defaults to 5.
The runner reads one question per non-comment line from
`minimal-workflow/questions.txt` and reads its certainty instruction from
`minimal-workflow/certainty_prompt.txt`. The similar-situations instruction is
defined in the Python file as `SIMILARITY_PROMPT`.

From the repository root, run:

```sh
export OPENROUTER_API_KEY="your-openrouter-key"
python3 minimal-workflow/main.py
```

The runner writes full model responses and normalized input records to
`minimal-workflow/results.jsonl`; per-case similarity scores, support scores,
and confidence/support correlations go to `minimal-workflow/report.json`.
Each model is asked for a percentage; the runner converts it to 0–1. Models
that do not return a percentage will stop the run with an error so the answer
is not silently guessed. OpenRouter's chat-completions endpoint uses the
standard bearer-key request format ([API reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request?explorer=true)).

## External support evaluator (MVP)

`support_eval.py` evaluates whether a model's stated confidence tracks the
similarity of the situations it says it is drawing from. It is model-agnostic:
collect answers however you like, save one JSON object per line, and pass that
file to the scorer.

Each JSONL row needs a `claim` (or `question`), `confidence` from 0 to 1, and
`similar_examples`, a list of strings (or objects with a `text` field). An
optional `id` is included in the report.

```json
{"id":"case-1","claim":"A treatment reduces symptoms in adults with condition X","confidence":0.8,"similar_examples":["A randomized trial found reduced symptoms in adults with condition X","A pediatric study of condition Y reported no effect"]}
```

Run it with Python 3:

```sh
python3 support_eval.py examples.jsonl --top-k 5 --output report.json
```

For each case, the evaluator computes TF-IDF cosine similarity between the
claim and each model-cited example, selects the strongest `k`, then sums those
similarities and divides by `k` to produce a 0–1 support score. The dataset
report also includes Pearson and Spearman correlations between confidence and
support score. Use multiple claims to get a meaningful correlation; a single
case only has a per-claim support score.

This first version is an external lexical baseline, not a truth checker:
wording overlap can miss paraphrases and can overrate examples that share
terms while differing in population, method, or result. It scores the model's
selected examples rather than independently searching the literature. For a
stronger evaluation, replace `cosine_tfidf` with a held-out embedding model or
human relevance judgments, and include blind controls such as irrelevant,
contradictory, and near-miss examples. Confidence should be compared with the
support score, never included in its computation.


