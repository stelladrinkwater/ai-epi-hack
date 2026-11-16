#!/usr/bin/env python3
"""Run the minimal confidence/support workflow against OpenRouter models."""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Run configuration. Set the key here or export OPENROUTER_API_KEY before run.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODELS = [
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.8-flash",
    "x-ai/grok-4.3",
    "z-ai/glm-5.3-flash",
    "stepfun/step-3.5-flash",
    "meta-llama/llama-4-scout",
]
K = 5
MIN_SECONDS_BETWEEN_MODEL_REQUESTS = 3.2  # Keeps each model below 20 requests/minute.
LAST_MODEL_REQUEST_AT = {}

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions.txt"
CERTAINTY_PROMPT_FILE = HERE / "certainty_prompt.txt"
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
    for attempt in range(5):
        wait = MIN_SECONDS_BETWEEN_MODEL_REQUESTS - (
            time.monotonic() - LAST_MODEL_REQUEST_AT.get(model, 0)
        )
        if wait > 0:
            time.sleep(wait)
        LAST_MODEL_REQUEST_AT[model] = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code != 429 or attempt == 4:
                raise RuntimeError(
                    f"OpenRouter returned HTTP {exc.code}: {detail}"
                ) from exc
            reset_at = exc.headers.get("X-RateLimit-Reset")
            retry_after = exc.headers.get("Retry-After")
            if reset_at:
                delay = max(1, float(reset_at) / 1000 - time.time() + 1)
            elif retry_after:
                delay = max(1, float(retry_after))
            else:
                delay = 10 * (2 ** attempt)
            print(f"[{model}] rate limited; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def parse_confidence(text: str) -> float:
    match = re.search(r"(?<!\d)(100(?:\.\d+)?|\d{1,2}(?:\.\d+)?)\s*%", text)
    if not match:
        # Some models put the score on a separate line without a percent sign.
        match = re.search(r"(?<!\d)(100(?:\.\d+)?|\d{1,2}(?:\.\d+)?)(?!\d)", text)
    if match:
        score = float(match.group(1))
        if 0 <= score <= 100:
            return score / 100
    raise ValueError(f"Could not find a percentage or standalone confidence score: {text!r}")


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

    expected_questions = {
        f"{model}:{index}": question
        for model in MODELS
        for index, question in enumerate(questions, start=1)
    }
    # Resume the newest compatible incomplete run to avoid repeating billed calls.
    output_file = None
    records = []
    for candidate in sorted(HERE.glob("results_*.jsonl"), key=lambda path: path.stat().st_mtime,
                            reverse=True):
        try:
            candidate_records = [json.loads(line) for line in
                                 candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        ids = [row.get("id") for row in candidate_records]
        candidate_report = HERE / candidate.name.replace("results_", "report_", 1).replace(
            ".jsonl", ".json"
        )
        if (candidate_records and len(ids) == len(set(ids))
                and all(case_id in expected_questions
                        and row.get("question") == expected_questions[case_id]
                        for row, case_id in zip(candidate_records, ids))
                and (len(ids) < len(expected_questions) or not candidate_report.exists())):
            output_file = candidate
            records = candidate_records
            break

    if output_file is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = HERE / f"results_{run_id}.jsonl"
        output_file.write_text("", encoding="utf-8")
        print(f"Writing completed cases to {output_file}", flush=True)
    else:
        print(f"Resuming {len(records)}/{len(expected_questions)} saved cases in {output_file}",
              flush=True)
    report_file = HERE / output_file.name.replace("results_", "report_", 1).replace(".jsonl", ".json")
    completed_ids = {row["id"] for row in records}
    for model in MODELS:
        for index, question in enumerate(questions, start=1):
            case_id = f"{model}:{index}"
            if case_id in completed_ids:
                continue
            print(f"[{model}] question {index}/{len(questions)}", flush=True)
            answer = openrouter_chat(model, question)
            confidence_prompt = (
                f"Question: {question}\nAnswer: {answer}\n\n"
                f"{certainty_prompt}\nGive your certainty as a percentage."
            )
            confidence = None
            certainty_response = ""
            confidence_parse_error = None
            for attempt in range(3):
                certainty_response = openrouter_chat(model, confidence_prompt)
                try:
                    confidence = parse_confidence(certainty_response)
                    break
                except ValueError as exc:
                    confidence_parse_error = str(exc)
                    if attempt < 2:
                        confidence_prompt = (
                            f"Question: {question}\nAnswer: {answer}\n\n"
                            "Report only your certainty as one number from 0 to 100 followed by %."
                        )
                        print(f"[{model}] couldn't parse confidence; requesting a numeric-only reply",
                              flush=True)
                    else:
                        print(f"[{model}] confidence unavailable after retries; continuing without it",
                              flush=True)
            similar_response = openrouter_chat(
                model, SIMILARITY_PROMPT.format(k=K, question=question, answer=answer)
            )
            record = {
                "id": case_id,
                "model": model,
                "question": question,
                "claim": f"{question}\nAnswer: {answer}",
                "answer": answer,
                "confidence": confidence,
                "certainty_response": certainty_response,
                "similar_examples": parse_examples(similar_response),
                "similarity_response": similar_response,
            }
            if confidence_parse_error and confidence is None:
                record["confidence_parse_error"] = confidence_parse_error
            records.append(record)
            completed_ids.add(case_id)
            with output_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Reuse the standalone lexical scorer and include its per-case scores and correlations.
    sys.path.insert(0, str(HERE.parent))
    from support_eval import analyze

    report = analyze(records, top_k=K)
    report["models"] = MODELS
    report["results_file"] = str(output_file)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved raw responses to {output_file}")
    print(f"Saved support scores and correlations to {report_file}")


if __name__ == "__main__":
    main()
