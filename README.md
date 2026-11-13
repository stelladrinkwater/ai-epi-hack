# ai-epi-hack
AI x Epistemics Research Sprint

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

## Track 1: Model Epistemics & Decision-Making Evals
Does the model understand how solid its claims are? Can you accurately measure its confidence in a decision and verify that against reality?
Can you measure the gap between what a model implies and what it will defend (e.g. overselling, "flip-flopping", misrepresented evidence)?
Which model behaviours matter for real decision quality, and does ranking well on an honesty benchmark predict better decision-making?
What does a benchmark need to stay meaningful over time? Which exploits make benchmarks gameable, and can you demonstrate and close one?
AI systems now recommend and increasingly make consequential decisions, and we do not currently have a great idea of the quality of their epistemics. Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified? Is it misrepresenting the solidity of what it tells you: overselling its own work, implying more than it will defend, retreating quietly under pushback? Evals are the cheapest external lever on model behaviour by shaping what labs optimise for.

## Core Question: "Does a model know how solid its claims are? Can its confidence in a recommendation be elicited and verified?"

### Note (on interpretation re: above): This is *not* about calibration of predictions, it's about measuring if confidence correlates to how well-supported the claim is.

## Eval Structure (Applying ML Principles):
1. Prompt model to define how certain it is of an answer to a question
    -> Simplify this to just certainty of statements? (i.e. not asking the model to answer a question then state certainty, instead just giving it a statement and ask "How certain is it that this statement is true"?)
2. Test certainty in a number of different ways:

    Best Idea Currently: With open weight models -> Look for a signal of confidence in the activations(?)

   ^ With open models we could try and test the question of "Is a model's reported certainty representative of how certain it actually predicts the answer to be?" (i.e. is the output misrepresenting the internal state)

*Thinking of ways people set their own internal confidence meters (**brainstorming** for applicable ideas):*

- If you apply a principle (or "law") to a many problems and in the past it has almost always given you the right answer, you will likely feel confident in your predictions when applying it to a new problem
  
- If a problem is very dissimilar to anything you have seen before, any prediction you make will likely be with a lower level of confidence

Insight from this: Can we get models to tell us what principles they are applying, if any? Can we ask for similar examples that support their confidence rating? Can we measure *how* similar each similar example is, then how many their are? Can some metric like this (e.g. get top 5 most similar past known examples, measure how similar they are, add the similarities to get a past-data-similarity-score) tell us something about confidence?

## Minimal Workflow (testing a minimal version of this idea first):

1. Ask model a question  
2. Ask model to state certainty of answer
3. Ask model to provide k (start with k=5, adjust later?) similar examples it is "drawing from"   
    - Also, to help generate more ideas, ask what principles are being used/"how it knows"/how it set the certainty level
    - Models will often just pull a percentage from an existing report. Find a way to filter this out.     
4. Use an external system to rate how similar each of these similar examples actually is
    - Create a similarity score: Some type of vector that compares the keywords/results in the similar examples(?)
5. Compare similarity score to model's stated confidence

Note: Used openrouter for steps 1-3
   
#### (Supporting Idea) To find hard questions fast:
1. Generate a list of questions
2. Ask the questions to a few different models
3. See where the output differs
    -> Use this subset of questions for eval-development

#### Ways to Extend support-score
- Allow chains of reasoning? Find a way to test these
- Find a way to audit math (logic and foundations) (i'm not phrasing this well)

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


