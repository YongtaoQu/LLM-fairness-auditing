#!/usr/bin/env python3
"""Bilingual (EN/MN) cloze experiment via OpenRouter. Python 3.8+, standard library only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

VERSION = "2.1"
HERE = Path(__file__).resolve().parent
LANGUAGES = {"en": "question_en", "mn": "question_mn_traditional"}
DEFAULT_BASE = "https://openrouter.ai/api/v1"
SEARCH_ENGINES = ("exa", "native", "auto", "none")


def search_tool_for(engine):
    """Build openrouter:web_search tool config, or None when search is disabled."""
    if engine == "none":
        return None
    if engine == "exa":
        return {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "exa",
                "mode": "auto",
                "max_results": 5,
                "max_uses": 3,
            },
        }
    # native / auto: avoid Exa-only fields that can 400 on provider-native search
    return {
        "type": "openrouter:web_search",
        "parameters": {"engine": engine},
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def api_secret():
    return os.environ.get("OPENROUTER_API_KEY", "")


def safe_json(value):
    text = json.dumps(value, ensure_ascii=False)
    secret = api_secret()
    return text.replace(secret, "[REDACTED]") if secret else text


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(safe_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, value, durable=True):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(safe_json(value) + "\n")
        stream.flush()
        if durable:
            os.fsync(stream.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sse_events(stream):
    """Decode SSE, including CRLF, comments and multi-line data fields."""
    fields = []
    for raw in stream:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if fields:
                yield "\n".join(fields)
                fields = []
        elif line.startswith("data:"):
            fields.append(line[5:].lstrip(" "))
    if fields:
        yield "\n".join(fields)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def worker():
    job = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    raw_path = Path(job["raw_path"])
    secret = api_secret()
    if not secret:
        raise KeyError("OPENROUTER_API_KEY")
    request = Request(job["endpoint"], method="POST",
                      data=json.dumps(job["payload"], ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": "Bearer " + secret,
                               "Content-Type": "application/json",
                               "Accept": "text/event-stream"})
    started = time.perf_counter()
    append_json(raw_path, {"kind": "started", "monotonic_start": time.monotonic(),
                           "started_at_utc": utc_now()}, durable=False)
    def emit(kind, **kwargs):
        append_json(raw_path, dict(kind=kind, elapsed_s=time.perf_counter() - started, **kwargs),
                    durable=False)
    try:
        # No retries: one trial corresponds to exactly one HTTP request.
        with build_opener(NoRedirect()).open(request, timeout=job["timeout_s"]) as response:
            emit("headers", http_status=response.status,
                 request_id=(response.headers.get("x-request-id")
                             or response.headers.get("x-openrouter-id")))
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                body = response.read(65536).decode("utf-8", errors="replace")
                emit("error", error_type="non_sse_response", message=body[:2000])
                return
            for data in sse_events(response):
                if data.strip() == "[DONE]":
                    emit("done")
                    break
                emit("response", data=json.loads(data))
        emit("finished")
    except HTTPError as exc:
        emit("error", error_type="http_error", http_status=exc.code,
             message=exc.read(65536).decode("utf-8", errors="replace")[:2000])
    except URLError as exc:
        error_type = "TimeoutError" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "URLError"
        emit("error", error_type=error_type, message=str(exc)[:2000])
    except Exception as exc:
        emit("error", error_type=type(exc).__name__, message=str(exc)[:2000])


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return "".join(parts)
    return ""


def reasoning_text(payload):
    """Extract reasoning text from OpenRouter delta/message fields."""
    if not isinstance(payload, dict):
        return ""
    chunks = []
    for key in ("reasoning", "reasoning_content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            chunks.append(value)
    details = payload.get("reasoning_details")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                chunks.append(detail.get("text") or detail.get("summary") or "")
            elif isinstance(detail, str):
                chunks.append(detail)
    return "".join(chunks)


def collect_annotations(payload, sources):
    if not isinstance(payload, dict):
        return
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        return
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        citation = annotation.get("url_citation") if annotation.get("type") == "url_citation" else annotation
        if not isinstance(citation, dict):
            citation = annotation
        url = citation.get("url") or annotation.get("url")
        key = url or json.dumps(annotation, sort_keys=True, ensure_ascii=False)
        sources[key] = annotation


def script_counts(text):
    # Character counts only, not a language classifier.
    return {"latin_chars": sum("LATIN" in unicodedata.name(c, "") for c in text),
            "cyrillic_chars": sum("CYRILLIC" in unicodedata.name(c, "") for c in text),
            "traditional_mongolian_chars": sum(0x1820 <= ord(c) <= 0x18AA for c in text)}


def summarize(events, forced_status=None, forced_elapsed=None):
    result = {"status": "incomplete", "started_at_utc": None,
              "end_to_end_s": None, "time_to_first_event_s": None,
              "time_to_first_answer_s": None, "answer": "", "reasoning": "",
              "finish_reason": None, "request_id": None, "http_status": None,
              "usage": {}, "search_evidence": "unknown", "sources": [],
              "search_info": [], "error_type": None, "error_message": None}
    usage, sources = {}, {}
    finished, explicit_error, saw_tool_call = False, False, False
    for event in events:
        kind = event.get("kind")
        if kind == "started":
            result["started_at_utc"] = event["started_at_utc"]
        elif kind == "headers":
            result["http_status"] = event["http_status"]
            result["request_id"] = event.get("request_id")
        elif kind == "finished":
            finished = True
            result["end_to_end_s"] = event["elapsed_s"]
        elif kind == "error":
            explicit_error = True
            result.update(status="error", error_type=event["error_type"],
                          error_message=event["message"], end_to_end_s=event["elapsed_s"])
            result["http_status"] = event.get("http_status", result["http_status"])
        elif kind == "response":
            elapsed, data = event["elapsed_s"], event["data"]
            if result["time_to_first_event_s"] is None:
                result["time_to_first_event_s"] = elapsed
            result["request_id"] = (data.get("id") or data.get("request_id")
                                    or result["request_id"])
            error = data.get("error")
            if error:
                explicit_error = True
                if isinstance(error, dict):
                    result.update(status="error",
                                  error_type=str(error.get("code") or "api_error"),
                                  error_message=error.get("message") or safe_json(error),
                                  end_to_end_s=elapsed)
                else:
                    result.update(status="error", error_type="api_error",
                                  error_message=str(error), end_to_end_s=elapsed)
                continue
            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            payload = delta if delta else message
            delta_text = content_text(payload.get("content"))
            if delta_text:
                if result["time_to_first_answer_s"] is None:
                    result["time_to_first_answer_s"] = elapsed
                result["answer"] += delta_text
            result["reasoning"] += reasoning_text(payload)
            collect_annotations(payload, sources)
            tool_calls = payload.get("tool_calls") or []
            if tool_calls:
                saw_tool_call = True
                result["search_info"].append({"tool_calls": tool_calls})
            reason = choice.get("finish_reason")
            if reason and reason != "null":
                result["finish_reason"] = reason
            # Streaming usage is cumulative; do not sum repeated frames.
            usage.update(data.get("usage") or {})
    result["usage"] = usage
    result["sources"] = list(sources.values())
    result["source_count"] = len(sources)
    if sources or saw_tool_call or usage.get("server_tool_use") or usage.get("plugins"):
        result["search_evidence"] = "observed"
    # Missing metadata means unknown, not proof that no search happened.
    if finished and not explicit_error:
        if result["finish_reason"] == "stop" and result["answer"].strip():
            result["status"] = "ok"
        elif result["finish_reason"] == "length":
            result["status"] = "truncated"
    if forced_status:
        result["status"] = forced_status
        result["end_to_end_s"] = forced_elapsed
    elif result["error_type"] in ("TimeoutError", "timeout"):
        result["status"] = "timeout"
    result["input_tokens"] = usage.get("input_tokens", usage.get("prompt_tokens"))
    result["output_tokens"] = usage.get("output_tokens", usage.get("completion_tokens"))
    result.update(script_counts(result["answer"]))
    return result


def payload_for(trial, args):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": trial["question"]}],
        "stream": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.thinking:
        reasoning = {"enabled": True}
        if args.reasoning_max_tokens:
            reasoning["max_tokens"] = args.reasoning_max_tokens
        payload["reasoning"] = reasoning
    tool = search_tool_for(args.search_engine)
    if tool is not None:
        payload["tools"] = [tool]
    return payload


def make_schedule(questions, passes, order_seed):
    rng = random.Random(order_seed)
    schedule = []
    for pass_number in range(1, passes + 1):
        block = []
        for question in questions:
            for language, field in LANGUAGES.items():
                block.append({"trial_id": "p{}-{}-{}".format(pass_number, question["id"], language),
                              "pass": pass_number, "question_id": question["id"],
                              "language": language, "question": question[field]})
        rng.shuffle(block)
        schedule.extend(block)
    return schedule


def run_trial(trial, args, endpoint, raw_path):
    job = {"payload": payload_for(trial, args), "endpoint": endpoint,
           "timeout_s": args.timeout, "raw_path": str(raw_path)}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_worker"],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=flags)
    process.stdin.write(json.dumps(job, ensure_ascii=False).encode("utf-8"))
    process.stdin.close()
    # monotonic is shared across processes on Python 3.8 Windows; perf_counter is not.
    startup, started = time.monotonic(), None
    forced_status, forced_elapsed = None, None
    try:
        while process.poll() is None:
            if started is None and raw_path.exists():
                with raw_path.open(encoding="utf-8") as stream:
                    first_line = stream.readline()
                try:
                    started = json.loads(first_line).get("monotonic_start")
                except ValueError:
                    pass  # The first line can briefly be incomplete during its write.
            elapsed = time.monotonic() - (started if started is not None else startup)
            deadline = args.timeout if started is not None else 30
            if elapsed >= deadline:
                forced_status = "timeout" if started is not None else "worker_start_error"
                process.kill()
                process.wait()
                forced_elapsed = elapsed if started is not None else None
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        forced_status = "interrupted"
        forced_elapsed = time.monotonic() - started if started is not None else None
    # Preserve complete frames even when termination interrupted the last line.
    events = []
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                break
    return summarize(events, forced_status, forced_elapsed)


CSV_FIELDS = ["trial_id", "pass", "question_id", "language", "model", "status",
              "end_to_end_s", "time_to_first_answer_s", "time_to_first_event_s",
              "search_evidence", "source_count", "input_tokens", "output_tokens",
              "finish_reason", "latin_chars", "cyrillic_chars", "traditional_mongolian_chars",
              "started_at_utc", "question", "answer", "http_status", "error_type",
              "error_message", "request_id", "manual_accuracy", "manual_answer_language"]


def export_csv(path, records):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = json.loads(safe_json(record))
            # Prevent Excel from interpreting generated text as formulas.
            for name, value in row.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    row[name] = "'" + value
            writer.writerow(row)
    temporary.replace(path)


def validate_base(base):
    candidate = (base or DEFAULT_BASE).strip()
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if (parsed.scheme != "https" or host not in ("openrouter.ai", "www.openrouter.ai")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or path not in ("", "/api/v1") or "{" in candidate or "<" in candidate):
        raise ValueError(
            "Set an OpenRouter HTTPS base URL, e.g. https://openrouter.ai/api/v1.")
    return DEFAULT_BASE.rstrip("/") + "/chat/completions"


def main():
    parser = argparse.ArgumentParser(
        description="EN/MN cloze experiment via OpenRouter: search, timing, repeats; stdlib only.")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.json")
    parser.add_argument("--run-dir", type=Path, default=HERE / "results/run")
    parser.add_argument("--model", default="qwen/qwen3.7-plus")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--ids", help="Run only these question IDs, e.g. Q01,Q02")
    parser.add_argument("--limit", type=int, help="Max new requests this run; useful for pilots")
    parser.add_argument("--order-seed", type=int, default=20261004)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Total output cap including thinking; default 16384")
    parser.add_argument("--reasoning-max-tokens", type=int, default=8192,
                        help="Thinking-only cap; default 8192 so the answer still has room")
    parser.add_argument("--timeout", type=float, default=600,
                        help="Per-request timeout in seconds; default 600")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--search-engine", choices=SEARCH_ENGINES, default="exa",
                        help="Web search backend: exa (default), native (provider search), auto, or none")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable deep thinking; off by default and recorded in the protocol")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the schedule only; do not call the API")
    args = parser.parse_args()
    if args.passes < 1 or args.max_tokens < 1 or args.timeout <= 0 or args.interval < 0:
        parser.error("passes, max_tokens, and timeout must be positive; interval cannot be negative.")
    if args.reasoning_max_tokens is not None and args.reasoning_max_tokens < 1:
        parser.error("--reasoning-max-tokens must be positive.")
    if args.thinking and args.reasoning_max_tokens and args.reasoning_max_tokens >= args.max_tokens:
        parser.error("--reasoning-max-tokens must be less than --max-tokens.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not 0 <= args.temperature < 2:
        parser.error("temperature must be in [0, 2).")
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question bank must be a non-empty JSON array.")
    identifiers = set()
    for question in questions:
        identifier = question.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError(
                "Question IDs must be unique and use only letters, digits, _ or -.")
        identifiers.add(identifier)
        for field in LANGUAGES.values():
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError("{} is missing {}".format(identifier, field))
    if args.ids:
        selected = set(args.ids.split(","))
        if selected - identifiers:
            raise ValueError("Unknown question IDs: " + str(selected - identifiers))
        questions = [q for q in questions if q["id"] in selected]
    schedule = make_schedule(questions, args.passes, args.order_seed)
    print("{} questions x 2 languages x {} passes = {} requests.".format(
        len(questions), args.passes, len(schedule)))
    print("Model {}; search_engine {}; thinking {}.".format(
        args.model, args.search_engine, args.thinking))
    if args.dry_run:
        print("Dry run OK; no API calls and no result files written.")
        return 0
    if not api_secret():
        raise ValueError("Set OPENROUTER_API_KEY first.")
    endpoint = validate_base(os.environ.get("OPENROUTER_BASE_HTTP_API_URL", DEFAULT_BASE))
    args.run_dir = args.run_dir.resolve()
    sample_payload = payload_for(schedule[0], args)
    recorded_payload = dict(sample_payload)
    recorded_payload["messages"] = [{"role": "user", "content": "[QUESTION]"}]
    search_tool = search_tool_for(args.search_engine)
    protocol = {"runner_version": VERSION, "provider": "openrouter",
                "model": args.model, "endpoint": endpoint,
                "passes": args.passes, "order_seed": args.order_seed,
                "interval_s": args.interval, "timeout_s": args.timeout,
                "request": recorded_payload,
                "search_engine": args.search_engine,
                "search_tool": search_tool,
                "questions_sha256": hashlib.sha256(json.dumps(questions, ensure_ascii=False,
                    sort_keys=True).encode("utf-8")).hexdigest(),
                "prompt_policy": "original question only; one new conversation per request",
                "model_seed": "provider default (not explicitly set)"}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.run_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["protocol"] != protocol:
            raise ValueError(
                "Question bank or settings changed; use a new --run-dir to avoid mixing runs.")
    else:
        write_json(manifest, {"created_at_utc": utc_now(), "protocol": protocol,
                              "python_version": sys.version, "schedule": schedule})
    result_path = args.run_dir / "results.jsonl"
    records = read_jsonl(result_path)
    seen = {record["trial_id"] for record in records}
    raw_dir = args.run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    pending = [trial for trial in schedule if trial["trial_id"] not in seen]
    orphans = [trial["trial_id"] for trial in pending if (raw_dir / (trial["trial_id"] + ".jsonl")).exists()]
    if orphans:
        raise ValueError(
            "Found unsynced raw logs {}; inspect them before retrying to avoid double billing.".format(
                orphans))
    export_csv(args.run_dir / "results.csv", records)
    print("Recorded {}; {} remaining. Failed trials are not retried automatically.".format(
        len(records), len(pending)))
    attempted = 0
    for trial in pending:
        if args.limit is not None and attempted >= args.limit:
            break
        raw_path = raw_dir / (trial["trial_id"] + ".jsonl")
        write_json(raw_dir / (trial["trial_id"] + ".request.json"), payload_for(trial, args))
        result = dict(trial, model=args.model, search_enabled=search_tool is not None,
                      forced_search_requested=False,
                      search_engine=args.search_engine,
                      search_tool=(search_tool or {}).get("type"),
                      raw_file=str(raw_path), **run_trial(trial, args, endpoint, raw_path))
        append_json(result_path, result)
        records.append(result)
        export_csv(args.run_dir / "results.csv", records)
        attempted += 1
        elapsed = (round(result["end_to_end_s"], 3)
                   if result["end_to_end_s"] is not None else "unknown")
        print("{}: {}; {} s; search_evidence {}".format(
            trial["trial_id"], result["status"], elapsed, result["search_evidence"]))
        if result["status"] == "interrupted":
            print("Saved interrupted trial.")
            return 130
        if result["status"] not in ("ok", "truncated"):
            print("Error saved; stopping. Re-run the same command to continue unfinished trials.")
            return 1
        if args.interval:
            time.sleep(args.interval)
    print("Results saved to {}".format(args.run_dir))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--_worker"]:
        worker()
    else:
        try:
            sys.exit(main())
        except (ValueError, OSError, KeyError) as exc:
            print("Error: {}".format(exc), file=sys.stderr)
            sys.exit(1)
