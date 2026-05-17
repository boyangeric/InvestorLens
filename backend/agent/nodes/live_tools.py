"""
Live Tools Node — OpenAI parallel tool calling for current market data + news.

Runs on the `hybrid_live` branch of the adaptive router. The flow is:

  1. A planner LLM call (gpt-4o-mini) sees the query, chat history, and a
     tool catalogue. It returns one OR MORE tool_calls in parallel —
     `get_market_quote` and `search_news` — via OpenAI's
     `parallel_tool_calls=True` native parallel-tool-call API.
  2. The tool invocations are dispatched concurrently in a ThreadPoolExecutor
     so latency stays at max(yfinance, tavily) rather than their sum.
  3. Results are written to `live_quotes` and `live_news`. The generator
     then reads BOTH retrieved chunks AND this live data and weaves them into
     one answer with distinct provenance.

We do NOT loop (no "agentic" multi-turn tool use here). One planning step,
one parallel execution, then on to the generator. Keeping a single round
makes the node deterministic in trace shape and easy to reason about.

Failures (Tavily unconfigured, yfinance can't resolve a ticker, etc.) return
empty lists rather than raise — the generator can still answer from the
retrieved chunks, and the trace records what was attempted vs what succeeded.
"""

import concurrent.futures
import json
import logging
import time

from openai.types.chat import ChatCompletionToolParam

from backend.agent.state import AgentState
from backend.agent.tools.market import MarketQuote, get_market_quote
from backend.agent.tools.news import NewsResult, search_news
from backend.agent.utils import (
    MINI_MODEL,
    compute_cost_usd,
    get_openai_client,
)

logger = logging.getLogger(__name__)


TOOL_SPECS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_market_quote",
            "description": (
                "Fetch a current live quote (price, day change, market cap, 52w "
                "range) for a single security. Use the .AX suffix for ASX-listed "
                "Australian companies (e.g. QAN.AX for Qantas, CBA.AX for "
                "Commonwealth Bank, BHP.AX for BHP, WBC.AX for Westpac)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol with exchange suffix, e.g. QAN.AX",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search recent financial news for a topic. Returns title, URL, "
                "snippet, and publication date for up to 5 recent articles. Use "
                "for 'recent news', 'announcements since', 'what happened to' "
                "style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language news query, e.g. 'Qantas profit downgrade'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Lookback window in days. Default 30.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


PLANNER_SYSTEM = (
    "You are the tool-planning step of a financial-document agent. The user's "
    "question requires CURRENT data the indexed reports cannot answer (live "
    "price, recent news, post-report state). Decide which tools to call.\n\n"
    "Rules:\n"
    "- Call at least one tool. You may call both in parallel.\n"
    "- For Australian companies use the .AX suffix (Qantas → QAN.AX, "
    "Commonwealth Bank → CBA.AX, BHP → BHP.AX, Westpac → WBC.AX, "
    "Macquarie → MQG.AX, Telstra → TLS.AX).\n"
    "- Keep news queries short and focused (the company name plus a topic "
    "keyword is usually enough)."
)


def _execute_tool_call(name: str, args: dict) -> tuple[str, dict, object | None, int, str | None]:
    """
    Run one tool call. Returns (name, args, result, duration_ms, error).
    Error is None on success or a short string on failure.
    """
    result: MarketQuote | list[NewsResult] | None
    start = time.time()
    try:
        if name == "get_market_quote":
            result = get_market_quote(args.get("ticker", ""))
        elif name == "search_news":
            result = search_news(
                args.get("query", ""),
                days=int(args.get("days", 30)),
            )
        else:
            return (name, args, None, 0, f"unknown tool: {name}")
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return (name, args, None, int((time.time() - start) * 1000), str(e))

    return (name, args, result, int((time.time() - start) * 1000), None)


def live_tools(state: AgentState) -> dict:
    """
    Plan + execute live tool calls in parallel; write results to state.

    Reads: query, chat_history
    Writes: live_quotes, live_news, node_trace
    """
    query = state["query"]
    chat_history = state.get("chat_history", [])

    logger.info("Live tools: planning for query — %s", query[:80])

    client = get_openai_client()
    planner_start = time.time()

    completion = client.chat.completions.create(
        model=MINI_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            *[{"role": m["role"], "content": m["content"]} for m in chat_history],
            {"role": "user", "content": query},
        ],
        tools=TOOL_SPECS,
        # "required" guarantees the planner picks at least one tool — we
        # already routed to hybrid_live so the answer needs live data.
        tool_choice="required",
        parallel_tool_calls=True,
        temperature=0,
    )

    planner_ms = int((time.time() - planner_start) * 1000)
    usage = completion.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0
    cost_usd = compute_cost_usd(MINI_MODEL, tokens_in, tokens_out)

    tool_calls = completion.choices[0].message.tool_calls or []

    # Dispatch every tool call to the threadpool simultaneously so wall time
    # is max(latencies) rather than sum.
    quotes: list[dict] = []
    news: list[dict] = []
    tool_trace_entries: list[dict] = []

    exec_start = time.time()
    if tool_calls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = []
            for tc in tool_calls:
                # The SDK union includes custom-tool calls (no .function attr).
                # We only registered function tools, so anything else is an
                # SDK quirk — skip rather than crash on attribute access.
                if tc.type != "function":
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                futures.append(pool.submit(_execute_tool_call, tc.function.name, args))

            for future in concurrent.futures.as_completed(futures):
                name, args, result, dur_ms, err = future.result()
                tool_trace_entries.append(
                    {"name": name, "args": args, "duration_ms": dur_ms, "ok": err is None, "error": err}
                )
                if err is not None or result is None:
                    continue
                if name == "get_market_quote":
                    quotes.append(result.model_dump())
                elif name == "search_news":
                    news.extend([r.model_dump() for r in result])

    exec_ms = int((time.time() - exec_start) * 1000)
    total_ms = planner_ms + exec_ms

    trace_entry = {
        "node": "live_tools",
        "status": f"{len(quotes)} quote(s), {len(news)} article(s)",
        "model": MINI_MODEL,
        "duration_ms": total_ms,
        "planner_ms": planner_ms,
        "tool_exec_ms": exec_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "tool_calls": tool_trace_entries,
        "market_count": len(quotes),
        "news_count": len(news),
    }

    node_trace = state.get("node_trace", []) + [trace_entry]
    logger.info(
        "Live tools: %d quote(s), %d article(s) in %dms (plan %dms, exec %dms)",
        len(quotes), len(news), total_ms, planner_ms, exec_ms,
    )

    return {
        "live_quotes": quotes,
        "live_news": news,
        # The grader is skipped for hybrid_live, so the generator would
        # otherwise see no docs. Promote retrieved_docs to relevant_docs so
        # the document side of the answer has context. The generator's prompt
        # treats them as supporting material, not the primary signal.
        "relevant_docs": state.get("retrieved_docs", []),
        "current_node": "live_tools",
        "node_trace": node_trace,
    }
