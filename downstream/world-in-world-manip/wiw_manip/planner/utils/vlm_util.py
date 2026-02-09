import asyncio
from typing import Iterable, Tuple, Dict, Any, List, Callable, Optional

async def _make_async_runner(
    concurrency: int = 4,
) -> Callable[[Callable[..., Any], Iterable[Tuple[tuple, dict]]], List[Any]]:
    """
    Returns an async function `runner(func, calls)` that executes many *blocking*
    calls concurrently in threads and preserves order.

    - concurrency: max concurrent calls
    - func: a blocking callable (e.g., your query_vlm)
    - calls: iterable of (args_tuple, kwargs_dict)

    Usage:
        runner = await _make_async_runner(concurrency=8)
        results = await runner(some_blocking_fn, [((arg1,arg2), {"kw": 3}), ...])
    """
    sem = asyncio.Semaphore(concurrency)

    async def runner(
        func: Callable[..., Any],
        calls: Iterable[Tuple[tuple, dict]],
    ) -> List[Any]:
        async def _one(a: tuple, kw: dict):
            async with sem:
                return await asyncio.to_thread(func, *a, **kw)

        tasks = [_one(a, kw) for (a, kw) in calls]
        return await asyncio.gather(*tasks)

    return runner


def run_concurrent_blocking(
    func: Callable[..., Any],
    calls: Iterable[Tuple[tuple, dict]],
    concurrency: int = 4,
    *,
    loop_already_running: bool = False,
) -> List[Any]:
    """
    Sync-friendly façade.
    - If you're NOT inside an event loop, it internally builds one and runs.
    - If you ARE already in an event loop (e.g., caller is async), set loop_already_running=True
      and call it like:
          results = asyncio.get_running_loop().run_until_complete( ... )  # not allowed
      Instead, directly use the async runner in that context:
          runner = await _make_async_runner(concurrency)
          results = await runner(func, calls)

    For normal sync code paths, just call:
        results = run_concurrent_blocking(fn, calls, concurrency=8)
    """
    async def _main():
        runner = await _make_async_runner(concurrency=concurrency)
        return await runner(func, calls)

    if loop_already_running:
        # Caller should not use this path; they should await _main() themselves.
        raise RuntimeError(
            "Event loop is already running. Use the async runner directly:\n"
            "  runner = await _make_async_runner(concurrency)\n"
            "  results = await runner(func, calls)"
        )
    return asyncio.run(_main())
