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
