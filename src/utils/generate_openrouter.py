"""
Generate with openrouter for a given model and a given set of prompts and kwargs.
"""

import asyncio
import os
import time

import fire
from dotenv import load_dotenv
from openrouter import OpenRouter

from src.utils.utils import dump_json, load_json

load_dotenv()

MAX_RETRIES = 3
DEFAULT_MAX_CONCURRENCY = 200
PROGRESS_LOG_INTERVAL_SECONDS = 5.0


async def call(client, model, prompt, **kwargs):
    """Make a single chat completion request with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.send_async(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2**attempt
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"All {MAX_RETRIES} attempts failed for prompt: {prompt[:80]}...")
                return None


async def generate(model, prompts, N, max_samples=None, max_concurrency=DEFAULT_MAX_CONCURRENCY, **kwargs):
    """Generate responses for all prompts concurrently.

    OpenRouter does not support an `n` parameter, so N completions per prompt
    are produced by issuing N independent requests in parallel.

    Args:
        model: OpenRouter model identifier (e.g. "openai/gpt-4o").
        prompts: List of prompt strings.
        N: Number of completions per prompt.
        max_samples: If set, only process the first max_samples prompts (useful for testing).
        max_concurrency: Maximum number of in-flight requests at any time.

    Returns:
        List (per prompt) of lists (length N) of response strings. Failed
        requests are represented as None.
    """
    if max_samples is not None:
        prompts = prompts[:max_samples]
    total = len(prompts) * N
    print(f"[openrouter] dispatching {total} requests to {model} ({len(prompts)} prompts x {N} completions, max_concurrency={max_concurrency})")

    state = {"started": 0, "done": 0, "last_log": time.monotonic()}
    sem = asyncio.Semaphore(max_concurrency)

    def maybe_log(force=False):
        now = time.monotonic()
        if force or (now - state["last_log"]) >= PROGRESS_LOG_INTERVAL_SECONDS:
            state["last_log"] = now
            in_flight = state["started"] - state["done"]
            print(f"[openrouter] progress: {state['done']}/{total} done, {in_flight} in-flight, {total - state['started']} queued")

    async def tracked_call(prompt):
        async with sem:
            state["started"] += 1
            maybe_log()
            result = await call(client, model, prompt, **kwargs)
            state["done"] += 1
            maybe_log()
            return result

    async with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        tasks = [tracked_call(prompt) for prompt in prompts for _ in range(N)]
        flat_responses = await asyncio.gather(*tasks)
    maybe_log(force=True)
    print(f"[openrouter] all {total} requests finished")
    return [flat_responses[i * N : (i + 1) * N] for i in range(len(prompts))]


def generate_sync(model, prompts, N=1, max_samples=None, max_concurrency=DEFAULT_MAX_CONCURRENCY, **kwargs):
    """Synchronous wrapper around generate."""
    return asyncio.run(
        generate(model, prompts, N, max_samples=max_samples, max_concurrency=max_concurrency, **kwargs)
    )


def main(model, prompts_json, N=1, max_samples=None, dump_path=None, **kwargs):
    """CLI entrypoint for generating responses via OpenRouter.

    Args:
        model: OpenRouter model identifier (e.g. "openai/gpt-4o").
        prompts_json: Path to a JSON file containing a list of prompt strings.
        N: Number of completions per prompt.
        max_samples: If set, only process the first N prompts (useful for testing).
        dump_path: If set, save responses to this JSON path.
    """
    prompts = load_json(prompts_json)
    list_of_responses = generate_sync(
        model, prompts, N, max_samples=max_samples, **kwargs
    )
    if dump_path is not None:
        dump_json(list_of_responses, dump_path)
    return list_of_responses


if __name__ == "__main__":
    fire.Fire(main)
