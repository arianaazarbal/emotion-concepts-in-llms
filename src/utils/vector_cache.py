"""Local read-through cache for ``results/emotion_vectors``.

The ``results/`` directory is typically either a symlink to slow persistent
storage or a local copy of it (either is fine). Readers (probe loading,
viewer directory scans) go through :func:`cached_results_root`, which copies
``results/emotion_vectors`` into ``./.cached/results/emotion_vectors`` on
first use and returns the cache root thereafter.

Writers (extract_emotion_vectors, denoise_emotion_vectors) must keep writing
to ``results/...`` directly. To refresh the cache after new vectors are
produced, delete ``./.cached/results/emotion_vectors``.
"""

import os
import shutil
import threading

CACHE_ROOT = ".cached/results"
SOURCE_SUBPATH = "results/emotion_vectors"
CACHED_SUBPATH = os.path.join(CACHE_ROOT, "emotion_vectors")
_lock = threading.Lock()
_populated = False


def cached_emotion_vectors_root() -> str:
    """Return the local cache dir for emotion vectors, populating it on first call.

    If ``./.cached/results/emotion_vectors`` is absent, copy-tree from
    ``results/emotion_vectors`` (following the symlink). Returns the cache path.
    Falls back to the source path if the source itself does not exist.
    """
    global _populated
    cached = CACHED_SUBPATH
    if _populated and os.path.isdir(cached):
        return cached
    with _lock:
        if os.path.isdir(cached):
            _populated = True
            return cached
        if not os.path.exists(SOURCE_SUBPATH):
            return SOURCE_SUBPATH
        os.makedirs(CACHE_ROOT, exist_ok=True)
        tmp = cached + ".tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        print(f"[vector_cache] copying {SOURCE_SUBPATH} -> {cached} (one-time)")
        shutil.copytree(SOURCE_SUBPATH, tmp, symlinks=False)
        os.rename(tmp, cached)
        _populated = True
        print(f"[vector_cache] cache ready at {cached}")
    return cached


def resolve_vector_subdir(relative_subpath: str) -> str:
    """Resolve a path under ``emotion_vectors/`` preferring the cache.

    ``relative_subpath`` is relative to ``emotion_vectors/`` (e.g.
    ``gemma2_9b_v0/start_at_50/denoised``). Falls back to the persistent
    source if the cache is missing the requested subtree, and top-ups the
    cache with that subtree so the next call is fast.
    """
    cached_root = cached_emotion_vectors_root()
    cached_path = os.path.join(cached_root, relative_subpath)
    if os.path.isdir(cached_path):
        return cached_path
    source_path = os.path.join(SOURCE_SUBPATH, relative_subpath)
    if not os.path.isdir(source_path):
        return cached_path
    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
    tmp = cached_path + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    print(f"[vector_cache] top-up {source_path} -> {cached_path}")
    shutil.copytree(source_path, tmp, symlinks=False)
    os.rename(tmp, cached_path)
    return cached_path

