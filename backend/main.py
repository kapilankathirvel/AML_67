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
    # Same source /query analyses. Calling load_data bare here used to take its
    # signature default, so with aml_data_source set the sidebar reported one
    # dataset (synthetic_alt: 1,710 txns / 294 customers) while every query
    # analysed another (synthetic: 2,002 / 270). Two numbers on one screen that
    # disagree is worse than either being wrong on its own.
    result = load_data(ctx, source=settings.aml_data_source)
    df = result.df if result.df is not None else ctx.df
    return {
        "row_count": 0 if df is None else len(df),
        "customer_count": 0 if ctx.customers is None else len(ctx.customers),
        "columns": [] if df is None else list(df.columns),
    }


def _pin_data_source(plan, dataset: str | None) -> str:
    """Force the load_data step onto one dataset, and say which.

    The planner emits load_data with empty params, so its signature default
    ('synthetic_alt') decides what gets analysed unless something overrides the
    step. That is fine locally and wrong for a deployment: 'synthetic_alt' is a
    different population from the labelled 'synthetic' set every published
    metric describes, so a demo left on the default answers questions about one
    dataset while README.md reports another.

    Overriding the step is the supported mechanism — evaluation/run_evaluation.py
    pins the source the same way, for the same reason.

    Deliberately applied AFTER planning and sourced from configuration rather
    than from the plan. plan_validator V14 blocks an LLM plan from reaching
    load_data's `source` at all; this is the other half of that decision, which
    is that choosing the dataset is the operator's call.
    """
    source = dataset or settings.aml_data_source
    for step in plan.steps:
        if step.tool == "load_data":
            step.params = {**step.params, "source": source}
    return source


@app.post("/query", response_model=AgentResponse)
def query(request: QueryRequest) -> AgentResponse:
    intent = parse_intent(request.query)
    # plan_query is build_plan plus an optional LLM planning step in front of
    # it; with settings.aml_llm_planner off (the default) it IS build_plan.
    plan = plan_query(intent)
    _pin_data_source(plan, request.dataset)
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
