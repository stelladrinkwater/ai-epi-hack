#!/usr/bin/env python3
"""Run the minimal confidence/support workflow against OpenRouter models."""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Run configuration. Set the key here or export OPENROUTER_API_KEY before run.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODELS = [
    "openai/gpt-4o-mini",
    # Add OpenRouter model IDs here, e.g. "anthropic/claude-3.5-sonnet"
]
K = 5

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions.txt"
CERTAINTY_PROMPT_FILE = HERE / "certainty_prompt.txt"
OUTPUT_FILE = HERE / "results.jsonl"
REPORT_FILE = HERE / "report.json"
SIMILARITY_PROMPT = (
    "List up to {k} specific situations or examples you drew on when answering "
    "the case below, ordered from most to least similar. Do not invent citations "
    "or present an illustrative example as a remembered source. Return only a "
    "numbered list, one situation per item. If you cannot identify any, say so "
    "rather than padding the list.\n\n"
    "Question: {question}\nAnswer: {answer}"
)


def openrouter_chat(model: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def parse_confidence(text: str) -> float:
    match = re.search(r"(?<!\d)(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%", text)
    if not match:
        raise ValueError(f"Could not find a percentage in certainty response: {text!r}")
    return float(match.group(1)) / 100


def parse_examples(text: str) -> list[str]:
    examples = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if cleaned and cleaned.lower() not in {"none", "n/a", "no relevant situations"}:
            examples.append(cleaned)
    return examples


def main() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "Set OPENROUTER_API_KEY in minimal-workflow/main.py or in your environment."
        )
    if not MODELS:
        raise SystemExit("Add at least one OpenRouter model ID to MODELS in main.py.")
    questions = [line.strip() for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    certainty_prompt = CERTAINTY_PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not questions:
        raise SystemExit(f"No questions found in {QUESTIONS_FILE}")
    if not certainty_prompt:
        raise SystemExit(f"Certainty prompt is empty: {CERTAINTY_PROMPT_FILE}")

    # One run creates a fresh JSONL and aggregate report.
    records = []
    for model in MODELS:
        for index, question in enumerate(questions, start=1):
            print(f"[{model}] question {index}/{len(questions)}", flush=True)
            answer = openrouter_chat(model, question)
            confidence_prompt = (
                f"Question: {question}\nAnswer: {answer}\n\n"
                f"{certainty_prompt}\nGive your certainty as a percentage."
            )
            certainty_response = openrouter_chat(model, confidence_prompt)
            confidence = parse_confidence(certainty_response)
            similar_response = openrouter_chat(
                model, SIMILARITY_PROMPT.format(k=K, question=question, answer=answer)
            )
            records.append({
                "id": f"{model}:{index}",
                "model": model,
                "question": question,
                "claim": f"{question}\nAnswer: {answer}",
                "answer": answer,
                "confidence": confidence,
                "certainty_response": certainty_response,
                "similar_examples": parse_examples(similar_response),
                "similarity_response": similar_response,
            })

    OUTPUT_FILE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                           encoding="utf-8")
    # Reuse the standalone lexical scorer and include its per-case scores and correlations.
    sys.path.insert(0, str(HERE.parent))
    from support_eval import analyze

    report = analyze(records, top_k=K)
    report["models"] = MODELS
    report["results_file"] = str(OUTPUT_FILE)
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved raw responses to {OUTPUT_FILE}")
    print(f"Saved support scores and correlations to {REPORT_FILE}")


if __name__ == "__main__":
    main()
