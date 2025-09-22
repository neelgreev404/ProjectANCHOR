"""
Project Anchor core: configuration, constants, and a BigQuery client.

Loads configuration (env + optional YAML/.env), exposes a typed `Config`,
defines `AllConstants` with defaults and validation, and provides a small
`smoke()` sanity check used by notebooks and scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

import json
import math
import os


# Try to load a local .env if it exists — dev convenience only; safe to ignore.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, that's fine; the OS env will do.
    pass


# Config helpers


def _load_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    """Try to parse a small config file.

    Prefers YAML if available; otherwise tries strict JSON as a decent fallback.
    Returns {} if the file is missing or parsing fails.
    """
    p = Path(path)
    if not p.exists():
        return {}

    # prefer YAML if we can import it — it's friendlier to hand-edit
    try:
        import yaml  # type: ignore

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # type: ignore
        return data if isinstance(data, dict) else {}
    except Exception:
        # if YAML isn't around, just see if it's valid JSON — good enough
        try:
            text = p.read_text(encoding="utf-8")
            return json.loads(text) if text.strip().startswith("{") else {}
        except Exception:
            return {}


def load_yaml_config() -> Dict[str, Any]:
    """Try to load a repo-level config file.

    Looks for `MSKG_CONFIG_YAML` first, else falls back to `config/mskg.yaml`.
    Returns {} if nothing is found or parse fails.
    """
    default_path = Path("config/mskg.yaml")
    configured = os.getenv("MSKG_CONFIG_YAML", str(default_path)).strip()
    path = configured or str(default_path)
    return _load_yaml_or_json(path)


# Environment bootstrap


def _bootstrap_yaml_env_if_available() -> None:
    """Pull a few env-style keys out of YAML and drop them into os.environ.

    Early in import so `AllConstants()` sees the same defaults `Config` would.
    If the OS env already has a value, we leave it alone.
    """
    try:
        yaml_cfg = load_yaml_config() or {}
        env_section = (
            yaml_cfg.get("env", {}) if isinstance(yaml_cfg.get("env", {}), dict) else {}
        )
        for key, value in env_section.items():
            if key not in os.environ or not str(os.environ.get(key, "")).strip():
                os.environ[key] = str(value)
    except Exception:
        # Don't fail import-time execution over optional configuration.
        pass


_bootstrap_yaml_env_if_available()


# Config object


@dataclass(frozen=True)
class Config:
    """Runtime configuration for Project Anchor with explicit typing and defaults.

    Values come from environment variables first, then YAML `env:` section,
    then hard-coded defaults as a last resort.
    """

    project_id: str
    region: str
    dataset_id: str
    staging_dataset_id: str
    quickstart: bool
    full_run: bool
    connection_name: Optional[str] = None

    # New knobs
    prompts_dir: str = "prompts"
    object_tables_enabled: bool = False
    gcs_images_prefix: str = ""
    gcs_pdfs_prefix: str = ""
    gcs_screens_prefix: str = ""

    @staticmethod
    def from_env() -> "Config":
        """Build a `Config` from env and YAML, with env taking precedence.

        YAML (if present) seeds defaults so templates like ${PROJECT_ID} expand
        correctly. Then we read the actual process env. Simple, predictable.
        """
        yaml_cfg = load_yaml_config() or {}
        env_section = (
            yaml_cfg.get("env", {}) if isinstance(yaml_cfg.get("env", {}), dict) else {}
        )

        # Pass 1: prime core identifiers so later ${VAR} expansions work.
        core_keys = ["PROJECT_ID", "REGION", "DATASET_ID", "STAGING_DATASET_ID"]
        for key in core_keys:
            candidate = str(env_section.get(key, "")).strip()
            if candidate and (
                key not in os.environ or not os.environ.get(key, "").strip()
            ):
                os.environ[key] = candidate

        # Pass 2: expand remaining YAML entries (so ${VAR} templates resolve) and set if missing.
        for key, raw_value in env_section.items():
            if key in core_keys:
                continue
            expanded = os.path.expandvars(str(raw_value))
            if key not in os.environ or not os.environ.get(key, "").strip():
                os.environ[key] = expanded

        def _get(key: str, default_value: str) -> str:
            # Environment has priority; otherwise YAML `env:`; else provided default.
            env_val = os.getenv(key, "").strip()
            if env_val:
                return env_val
            yaml_val = str(env_section.get(key, "")).strip()
            return yaml_val if yaml_val else default_value

        project_id = _get("PROJECT_ID", "your-project")
        region = _get("REGION", "US")
        dataset_id = _get("DATASET_ID", "mskg_demo")
        staging_dataset_id = _get("STAGING_DATASET_ID", "mskg_staging")

        # Flags (treat any non-empty non-"true" string as false).
        quickstart = (
            os.getenv(
                "QUICKSTART", str(env_section.get("QUICKSTART", "true")).strip()
            ).lower()
            == "true"
        )
        full_run = (
            os.getenv(
                "FULL_RUN", str(env_section.get("FULL_RUN", "false")).strip()
            ).lower()
            == "true"
        )

        # Connection: default to {project}.{region}.gemini unless explicitly overridden.
        default_conn = f"{project_id}.{region}.gemini"
        connection_name = os.getenv("BQ_CONNECTION_NAME", "").strip() or str(
            env_section.get("BQ_CONNECTION_NAME", default_conn)
        )

        # New knobs
        prompts_dir = _get("PROMPTS_DIR", "prompts")
        object_tables_enabled = _get("OBJECT_TABLES_ENABLED", "false").lower() == "true"
        gcs_images_prefix = _get("GCS_IMAGES_URI_PREFIX", "")
        gcs_pdfs_prefix = _get("GCS_PDFS_URI_PREFIX", "")
        gcs_screens_prefix = _get("GCS_SCREENS_URI_PREFIX", "")

        return Config(
            project_id=project_id,
            region=region,
            dataset_id=dataset_id,
            staging_dataset_id=staging_dataset_id,
            quickstart=quickstart,
            full_run=full_run,
            connection_name=connection_name,
            prompts_dir=prompts_dir,
            object_tables_enabled=object_tables_enabled,
            gcs_images_prefix=gcs_images_prefix,
            gcs_pdfs_prefix=gcs_pdfs_prefix,
            gcs_screens_prefix=gcs_screens_prefix,
        )

    # Convenience helpers

    def table(self, name: str) -> str:
        """Return a fully-qualified table name for the primary dataset."""
        return f"`{self.project_id}.{self.dataset_id}.{name}`"

    def staging_table(self, name: str) -> str:
        """Return a fully-qualified table name for the staging dataset."""
        return f"`{self.project_id}.{self.staging_dataset_id}.{name}`"

    def connection_id_short(self) -> str:
        """Return short-form connection id: project.location.connection.

        If a full resource path is provided via env (projects/.../connections/...),
        it is down-converted to short form for places that expect it.
        """
        base = (
            self.connection_name or f"{self.project_id}.{self.region}.gemini"
        ).strip()

        if base.startswith("projects/") and "/connections/" in base:
            # projects/{p}/locations/{loc}/connections/{name}
            parts = base.split("/")
            try:
                proj = parts[1]
                loc = parts[3]
                name = parts[5]
                return f"{proj}.{loc}.{name}"
            except Exception:
                # Fall back to a conventional default if parsing fails.
                return f"{self.project_id}.{self.region}.gemini"

        return base

    def connection_id_full(self) -> str:
        """Return the full resource connection path (projects/.../connections/...).

        Useful for OBJ.MAKE_REF and any API surface that requires a full path.
        """
        short = self.connection_id_short()
        if short.startswith("projects/"):
            return short

        try:
            proj, loc, name = short.split(".", 2)
            return f"projects/{proj}/locations/{loc}/connections/{name}"
        except Exception:
            # Fall back to a conventional default in case of unexpected formats.
            return (
                f"projects/{self.project_id}/locations/{self.region}/connections/gemini"
            )

    def assert_connection_region_ok(self) -> None:
        """Ensure the connection location matches the configured REGION.

        This guard catches a common pitfall where a connection is in a different
        location than the dataset/queries, leading to confusing runtime errors.
        """
        short = self.connection_id_short()
        try:
            _, loc, _ = short.split(".", 2)
        except Exception:
            loc = self.region

        assert (
            loc.upper() == self.region.upper()
        ), f"Connection location {loc} must match REGION {self.region}"


# BigQuery client helper


def get_bq_client(config: Config) -> Any:
    """Spin up a BigQuery client for this project/region.

    Nothing fancy here — just makes sure we don't accidentally query
    the wrong region.
    """
    from google.cloud import bigquery  # type: ignore  # local import keeps deps optional

    client = bigquery.Client(
        project=config.project_id,
        location=config.region,
    )

    # TODO: consider caching the client if we call this often in CLI tools
    return client


# Constants (with env/YAML overrides)


def _default_model(project_env: str, dataset_env: str, name: str) -> str:
    """Return a dataset-local remote model name if project/dataset are known."""
    proj = os.getenv("PROJECT_ID", project_env)
    ds = os.getenv("DATASET_ID", dataset_env)
    return f"{proj}.{ds}.{name}"


def _get_env_or_fallback(name: str, dataset_default: str, fallback: str) -> str:
    """Resolve a model/identifier setting with safe environment expansion.

    Resolution order:
    1) If env var `name` is set, expand ${VAR}s; if unresolved placeholders
       remain, prefer `dataset_default` (when available), otherwise `fallback`.
    2) Otherwise, prefer `dataset_default` if provided, else `fallback`.
    """
    env_raw = os.getenv(name, "").strip()
    if env_raw:
        try:
            # Expand common placeholders including general $VAR patterns.
            expanded = os.path.expandvars(env_raw)
            expanded = (
                expanded.replace("${PROJECT_ID}", os.getenv("PROJECT_ID", ""))
                .replace("${DATASET_ID}", os.getenv("DATASET_ID", ""))
                .replace("${REGION}", os.getenv("REGION", ""))
            )
            if "${" in expanded or "$" in expanded:
                # Unresolved placeholders remain — use safer defaults.
                return dataset_default if dataset_default else fallback
            return expanded
        except Exception:
            # On any expansion error, return the raw value rather than crashing.
            return env_raw

    # If no env override, prefer a dataset-local default when possible.
    if dataset_default:
        return dataset_default
    return fallback


@dataclass(frozen=True)
class EmbeddingIndexing:
    """Embedding and vector index configuration."""

    # Prefer dataset-local remote models; fall back to public model ids.
    text_model: str = _get_env_or_fallback(
        "TEXT_EMBED_MODEL",
        _default_model("your-project", "mskg_demo", "text_embedding_model"),
        "bq/embedding-text-1",
    )
    image_model: str = _get_env_or_fallback(
        "IMAGE_EMBED_MODEL",
        _default_model("your-project", "mskg_demo", "image_embedding_model"),
        "bq/embedding-image-1",
    )
    text_vector_dim: int = int(os.getenv("TEXT_VECTOR_DIM", "768"))
    image_vector_dim: int = int(os.getenv("IMAGE_VECTOR_DIM", "1024"))
    vector_k: int = int(os.getenv("VECTOR_K", "40"))
    vector_search_timeout_ms: int = int(os.getenv("VECTOR_SEARCH_TIMEOUT_MS", "2000"))
    index_refresh_cron: str = os.getenv("INDEX_REFRESH_CRON", "0 3 * * *")
    index_growth_thresh_pct: int = int(os.getenv("INDEX_GROWTH_THRESH_PCT", "15"))


@dataclass(frozen=True)
class ChunkingLimits:
    """Chunking and context limits used during enrichment and answers."""

    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "800"))
    max_image_side_px: int = int(os.getenv("MAX_IMAGE_SIDE_PX", "1024"))
    max_objects_per_run: int = int(os.getenv("MAX_OBJECTS_PER_RUN", "200000"))
    topk_context: int = int(os.getenv("TOPK_CONTEXT", "12"))


@dataclass(frozen=True)
class Validators:
    """Validator thresholds and limits."""

    validator_conf_thresh: float = float(os.getenv("VALIDATOR_CONF_THRESH", "0.65"))
    contradiction_margin: float = float(os.getenv("CONTRADICTION_MARGIN", "0.20"))
    max_validators_per_item: int = int(os.getenv("MAX_VALIDATORS_PER_ITEM", "5"))


@dataclass(frozen=True)
class Forecasting:
    """Forecast function and knobs."""

    forecast_func: str = os.getenv("FORECAST_FUNC", "AI.FORECAST")
    forecast_horizon_days: int = int(os.getenv("FORECAST_HORIZON_DAYS", "30"))
    forecast_backtest_splits: int = int(os.getenv("FORECAST_BACKTEST_SPLITS", "3"))


@dataclass(frozen=True)
class BudgetsSLA:
    """Latency and cost guardrails for demos and interactive use."""

    latency_p95_ms: int = int(os.getenv("LATENCY_P95_MS", "3000"))
    vector_latency_p95_ms: int = int(os.getenv("VECTOR_LATENCY_P95_MS", "500"))
    max_gb_scanned_per_demo: int = int(os.getenv("MAX_GB_SCANNED_PER_DEMO", "5"))
    max_bq_dml_per_day: int = int(os.getenv("MAX_BQ_DML_PER_DAY", "3"))


@dataclass(frozen=True)
class EvaluationCaching:
    """Evaluation targets and lightweight caching controls."""

    holdout_size: int = int(os.getenv("HOLDOUT_SIZE", "200"))
    recall_target: float = float(os.getenv("RECALL_TARGET", "0.95"))
    precision_at_k_target: float = float(os.getenv("PRECISION_AT_K_TARGET", "0.85"))
    mrr_target: float = float(os.getenv("MRR_TARGET", "0.80"))
    hot_query_topn: int = int(os.getenv("HOT_QUERY_TOPN", "25"))
    neighbor_precomp_k: int = int(os.getenv("NEIGHBOR_PRECOMP_K", "64"))
    cache_ttl_min: int = int(os.getenv("CACHE_TTL_MIN", "60"))
    prompt_ab_split_frac: float = float(os.getenv("PROMPT_AB_SPLIT_FRAC", "0.5"))
    max_mv_refreshes_per_day: int = int(os.getenv("MAX_MV_REFRESHES_PER_DAY", "4"))
    max_validation_sample_frac: float = float(
        os.getenv("MAX_VALIDATION_SAMPLE_FRAC", "0.25")
    )

    # Hybrid scoring weights (should sum to 1.0)
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.6"))
    hybrid_beta: float = float(os.getenv("HYBRID_BETA", "0.2"))
    hybrid_gamma: float = float(os.getenv("HYBRID_GAMMA", "0.1"))
    hybrid_delta: float = float(os.getenv("HYBRID_DELTA", "0.1"))


@dataclass(frozen=True)
class Orchestration:
    """Incidental orchestration settings (e.g., reproducible sampling)."""

    seed: int = int(os.getenv("SEED", "42"))


@dataclass(frozen=True)
class AllConstants:
    """Aggregate view of all tunable constants with validation helpers."""

    embed: EmbeddingIndexing = EmbeddingIndexing()
    chunk: ChunkingLimits = ChunkingLimits()
    validators: Validators = Validators()
    forecast: Forecasting = Forecasting()
    budgets: BudgetsSLA = BudgetsSLA()
    eval: EvaluationCaching = EvaluationCaching()
    orch: Orchestration = Orchestration()

    def validate(self) -> None:
        """Quick sanity sweep for coherence across constants.

        Not exhaustive; meant to catch obvious footguns before queries run.
        """
        # embedding dims and search breadth should be positive/reasonable
        assert self.embed.text_vector_dim > 0
        assert self.embed.image_vector_dim > 0
        assert 1 <= self.embed.vector_k <= 1000

        # context shouldn't exceed retrieval breadth
        assert self.chunk.topk_context > 0
        assert self.embed.vector_k >= self.chunk.topk_context

        # validator thresholds should be sane
        assert 0 < self.validators.validator_conf_thresh <= 1
        assert 0 <= self.validators.contradiction_margin <= 1

        # budget/eval targets shouldn't be degenerate
        assert self.budgets.max_gb_scanned_per_demo > 0
        assert 0 < self.eval.recall_target <= 1
        assert 0 < self.eval.precision_at_k_target <= 1
        assert 0 < self.eval.mrr_target <= 1
        assert 0 < self.eval.prompt_ab_split_frac < 1

        # hybrid weights: each in [0,1] and roughly sum to 1.0
        weights = (
            self.eval.hybrid_alpha,
            self.eval.hybrid_beta,
            self.eval.hybrid_gamma,
            self.eval.hybrid_delta,
        )
        for w in weights:
            assert 0.0 <= w <= 1.0

        assert math.isclose(sum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-9)


# Smoke test


def smoke() -> int:
    """Lightweight sanity check for local environments.

    Ensures configuration loads, constants validate, connection region matches,
    and table naming helpers work. Returns 0 on success.
    """
    cfg = Config.from_env()
    AllConstants().validate()

    # Exercise helpers (no-ops but ensure they resolve cleanly).
    _ = cfg.table("text_emb")
    _ = cfg.staging_table("tmp")
    cfg.assert_connection_region_ok()

    return 0


if __name__ == "__main__":
    raise SystemExit(smoke())
