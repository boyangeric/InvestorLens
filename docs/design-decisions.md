# Design Decisions

## 1. Single Agent over Multi-Agent

**Decision**: One agent with adaptive routing and multiple tools.
**Why**: The task doesn't require specialized reasoning from separate agents. Simpler, cheaper, easier to evaluate.

## 2. Adaptive Retrieval

**Decision**: Agent chooses retrieval strategy per query (semantic/keyword/extract/compare).
**Why**: Different queries need fundamentally different approaches. "What is the MER?" needs keyword search, not semantic similarity.

## 3. Self-Assessment over LLM-as-Judge

**Decision**: Generator produces confidence score in the same call.
**Why**: One LLM call instead of three. The model's self-assessment correlates well with answer quality for grounded generation tasks.

## 4. MCP Server

**Decision**: Expose tools via Model Context Protocol.
**Why**: Interoperability — any MCP client can use the tools without custom integration.

## 5. Human-in-the-Loop for Risk Flagging

**Decision**: Risk flagger pauses for human approval via LangGraph interrupt.
**Why**: Enterprise requirement for financial services — automated risk assessments need human oversight.

## 6. Prompt Versioning via YAML

**Decision**: All prompts in versioned YAML files.
**Why**: Enables iteration, A/B testing, and audit trails without code changes.

## 7. Behavioural Eval Suite over Unit Tests of Prompts

**Decision**: Evaluate the agent end-to-end against a fixed question set with
per-metric pass rates and a CI gate, rather than unit-testing prompt outputs.
**Why**: Prompt unit tests pin behaviour to a single golden string and break
on any reasonable rewording. The eval suite asserts the *properties* that
matter — moderator blocked the injection, router picked an allowed strategy,
every citation points to a retrieved chunk, faithfulness audit passed — which
survives prompt edits and catches regressions that touch graph topology, not
just text. Five categories map to five distinct graph branches, so the eval
doubles as documentation of the architecture's failure modes.

## 8. CRAG-style Two-Gate Grounding

**Decision**: Two independent gates decide whether to answer from the corpus
or fall back to general knowledge — (1) a strict relevance grader before
generation, (2) a faithfulness audit after generation. Either gate failing
routes to `general_generator`, which produces an explicitly ungrounded
answer with no confidence score and a different UI label.
**Why**: A single gate is brittle. The grader can let through topic-adjacent
chunks that don't actually answer the query, leading to citation
hallucination (real cite, fact from training data). The post-gen audit
catches that case. Two cheap LLM calls beat one wrong confident answer in a
financial context, where a fabricated citation is the worst possible
failure mode.
