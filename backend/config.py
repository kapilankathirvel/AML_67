from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    # How long to keep the Ollama model in VRAM after the last call.
    # Prevents cold-start reload on every request (default Ollama idle = 5m).
    ollama_keep_alive: str = "30m"
    # Per-call LLM timeout in seconds.  Hosted APIs (Gemini/Groq/OpenAI) are
    # fast and rarely need more than 10s.  Local Ollama with a 14B model may
    # take 15-30s for longer prompts, so allow more headroom.
    llm_timeout_seconds: int = 60
    # Max number of flags per response that get an LLM-polished explanation
    # (top N by risk_score). Bounds worst-case per-request LLM call count
    # regardless of how many entities end up HIGH-risk — a full_analysis run
    # can produce 20+ HIGH flags, and at several seconds per call (especially
    # on local Ollama, which has no rate limit to fail fast on) that multiplies
    # past the frontend's request timeout. The rest still get the accurate
    # template-based explanation, just not LLM-rewritten prose.
    llm_polish_max_flags: int = 5

    aml_use_mocks: bool = True
    aml_dataset_path: str = "data/sample/aml_sample.csv"
    aml_api_base_url: str = "http://localhost:8000"

    # Which dataset /query analyses. Applied to the load_data step in
    # backend/main.py, because the planner emits load_data with empty params
    # and overriding that step is the supported way to pin it — the same thing
    # evaluation/run_evaluation.py does for exactly the same reason.
    #
    # Defaults to load_data's own default, so behaviour is unchanged unless
    # somebody sets it. A DEPLOYMENT should set it to "synthetic": that is the
    # labelled 2,002-txn / 270-customer set every published metric is computed
    # against, whereas "synthetic_alt" is a different population entirely
    # (1,710 txns / 294 customers, different customer IDs). A demo running on
    # one while README.md reports the other invites a reasonable visitor to
    # conclude the numbers were invented.
    #
    # This is an operator setting and never a model-chosen one. plan_validator
    # V14 exists precisely to stop an LLM plan reaching load_data's `source`;
    # pinning it from configuration is the other half of the same decision.
    # A plan may choose how to analyse, not which dataset the product runs on.
    aml_data_source: str = "synthetic_alt"

    # Let the LLM choose which tools to run (backend/agent/llm_planner.py),
    # validated against backend/agent/plan_validator.py and falling back to the
    # deterministic planner on any failure.
    #
    # Defaults to False deliberately. There is no tests/conftest.py: each test
    # file stubs the LLM per-module, so a default of True would let the test
    # suite and the evaluation harness make real network calls against the .env
    # keys. Off by default means this repo's default behaviour is byte-identical
    # to what it was before the LLM planner existed, and the eval harness stays
    # reproducible without depending on anyone remembering to stub it.
    aml_llm_planner: bool = False

    # Let the model revise the REMAINING steps mid-run after observing what the
    # steps so far produced (backend/agent/replanner.py). This is the
    # observe -> decide -> act loop; without it the model commits to a plan
    # before any data is loaded.
    #
    # Off by default for the same reason as aml_llm_planner — no
    # tests/conftest.py means a default of on would let the suite issue real
    # network calls — and for one more: each re-plan is an extra LLM round trip
    # inside the request, and live queries already approach the frontend's 60s
    # timeout.
    aml_llm_replanner: bool = False


settings = Settings()
