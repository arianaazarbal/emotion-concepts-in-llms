"""Print an Inspect EvalLog to stdout in a transcript-friendly format.

Usage:
    uv run python -m scripts.view_inspect_log --path path/to/run.eval
    uv run python -m scripts.view_inspect_log --path ... --sample lcbhard_0
    uv run python -m scripts.view_inspect_log --path ... --max_chars 2000
"""

import fire

from inspect_ai.log import read_eval_log


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n… [{len(text) - max_chars} chars elided] …\n" + text[-half:]


def _print_sample(sample, max_chars: int):
    print("=" * 80)
    print(f"sample: {sample.id}")
    meta = sample.metadata or {}
    if "task_id" in meta:
        print(f"task_id: {meta['task_id']}   impossible_type: {meta.get('impossible_type')}")
    for name, score in (sample.scores or {}).items():
        print(f"score[{name}]: {score.value}")
        if score.answer:
            print(f"  answer (first 120 chars): {score.answer[:120]!r}")
    print("-" * 80)
    for i, msg in enumerate(sample.messages or []):
        role = getattr(msg, "role", type(msg).__name__)
        text = getattr(msg, "text", None) or ""
        print(f"[{i}] {role}:")
        print(_truncate(text, max_chars))
        print()
    if sample.output and getattr(sample.output, "completion", None):
        print("-" * 80)
        print("final completion:")
        print(_truncate(sample.output.completion, max_chars))


def main(path: str, sample: str = None, max_chars: int = 1500):
    """Dump an Inspect .eval log to stdout.

    Args:
        path: Path to the .eval (or .json) log file.
        sample: Optional sample id filter (e.g. "lcbhard_0"); default = all.
        max_chars: Cap per message/completion. 0 for no cap.
    """
    log = read_eval_log(path)
    header = log.eval
    print(f"task: {header.task}")
    print(f"model: {header.model}")
    print(f"samples: {len(log.samples or [])}")
    if log.results and log.results.scores:
        for s in log.results.scores:
            for metric_name, metric in (s.metrics or {}).items():
                print(f"metric[{s.name}.{metric_name}] = {metric.value}")
    print()
    for sm in log.samples or []:
        if sample is not None and str(sm.id) != str(sample):
            continue
        _print_sample(sm, max_chars)


if __name__ == "__main__":
    fire.Fire(main)
