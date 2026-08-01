"""
FastAPI app. Owner: Track A. Endpoints per docs/CONTRACTS.md Contract 1 HTTP surface.

/query runs the full intent_parser -> planner -> executor -> narrator pipeline.
/dataset/summary calls load_data directly (works against mocks; swaps to Track
B's real loader automatically once AML_USE_MOCKS=0 and backend/tools/data_loader.py
exists, since it goes through the same registry).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import registry
from backend.agent.executor import run_plan
from backend.agent.intent_parser import parse_intent
from backend.agent.llm_planner import plan_query, record_executed_plan
from backend.config import settings
from backend.schemas import AgentResponse
from backend.tools.base import ToolContext

app = FastAPI(title="AML Suspicious Activity Detection Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# plan_id -> AgentResponse, populated by /query, read back by /plan/{plan_id}
_RUN_CACHE: dict[str, AgentResponse] = {}


class QueryRequest(BaseModel):
    query: str
    dataset: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_available": bool(settings.gemini_api_key or settings.openai_api_key),
        "mocks": settings.aml_use_mocks,
    }


@app.get("/dataset/summary")
def dataset_summary() -> dict:
    tools = registry.load_tools(use_mocks=settings.aml_use_mocks)
    load_data = tools.get("load_data")
    if load_data is None:
        raise HTTPException(status_code=501, detail="load_data tool not available")

    ctx = ToolContext(df=None, customers=None, intent=None, artifacts={})
    result = load_data(ctx)
    df = result.df if result.df is not None else ctx.df
    return {
        "row_count": 0 if df is None else len(df),
        "customer_count": 0 if ctx.customers is None else len(ctx.customers),
        "columns": [] if df is None else list(df.columns),
    }


@app.post("/query", response_model=AgentResponse)
def query(request: QueryRequest) -> AgentResponse:
    intent = parse_intent(request.query)
    # plan_query is build_plan plus an optional LLM planning step in front of
    # it; with settings.aml_llm_planner off (the default) it IS build_plan.
    plan = plan_query(intent)
    response = run_plan(intent, plan)
    # After run_plan, so the trace records the executor's own mid-run
    # re-planning rather than only what was planned up front.
    record_executed_plan(plan)
    _RUN_CACHE[plan.plan_id] = response
    return response


@app.get("/plan/{plan_id}", response_model=AgentResponse)
def get_plan(plan_id: str) -> AgentResponse:
    if plan_id not in _RUN_CACHE:
        raise HTTPException(status_code=404, detail=f"no cached run for plan_id={plan_id}")
    return _RUN_CACHE[plan_id]
