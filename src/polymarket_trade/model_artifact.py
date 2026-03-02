from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

from polymarket_trade.config import TradeConfig

REQUIRED_MODEL_FILES = (
    "logistic.joblib",
    "lightgbm.joblib",
    "catboost.joblib",
    "stacking_meta.joblib",
    "meta.json",
)
MANIFEST_FILE = "model_manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_complete(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_MODEL_FILES)


def _validate_optimal_manifest(cfg: TradeConfig, model_dir: Path) -> None:
    if not cfg.model_artifact_require_optimal:
        return
    manifest_path = model_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise RuntimeError(f"缺少 {MANIFEST_FILE}，无法确认是否最优模型")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("is_best_from_opt", False)):
        raise RuntimeError("模型 manifest 未标记为最优模型（is_best_from_opt=false）")
    expected_model = (cfg.model_artifact_expected_model_name or "").strip()
    if expected_model:
        actual_model = str(manifest.get("model_name", "")).strip()
        if actual_model != expected_model:
            raise RuntimeError(f"模型类型不匹配: expected={expected_model}, actual={actual_model}")
    expected_opt_sha = (cfg.model_artifact_expected_opt_sha256 or "").strip().lower()
    if expected_opt_sha:
        actual_opt_sha = str(manifest.get("opt_sha256", "")).strip().lower()
        if actual_opt_sha != expected_opt_sha:
            raise RuntimeError("模型来源 opt 文件哈希不匹配，拒绝加载")


def _extract_to_staging(zip_path: Path, staging_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(staging_dir)
    # 兼容两种压缩结构：直接是文件，或包含一层 fixed_15m_stacking 目录
    direct_ok = all((staging_dir / name).exists() for name in REQUIRED_MODEL_FILES)
    if direct_ok:
        return staging_dir
    nested = staging_dir / "fixed_15m_stacking"
    nested_ok = all((nested / name).exists() for name in REQUIRED_MODEL_FILES)
    if nested_ok:
        return nested
    raise RuntimeError("模型压缩包内容不合法，未找到必需模型文件")


def _download_zip(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=30, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"模型下载失败: HTTP {r.status_code}")
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def ensure_model_artifact(root: Path, cfg: TradeConfig) -> None:
    model_dir = root / cfg.model_dir
    if _model_complete(model_dir):
        if not cfg.model_artifact_enabled:
            return
        try:
            _validate_optimal_manifest(cfg, model_dir)
            return
        except Exception:
            # 已有模型不满足最优校验时，继续走下载覆盖流程
            pass

    if not cfg.model_artifact_enabled:
        return

    url = (cfg.model_artifact_url or "").strip()
    if not url:
        raise RuntimeError("model_artifact.enabled=true 但未配置 model_artifact.url")

    with tempfile.TemporaryDirectory(prefix="predicta_model_") as td:
        tmp_dir = Path(td)
        zip_path = tmp_dir / "model.zip"
        _download_zip(url, zip_path)

        expected_sha = (cfg.model_artifact_sha256 or "").strip().lower()
        if expected_sha:
            actual_sha = _sha256_file(zip_path).lower()
            if actual_sha != expected_sha:
                raise RuntimeError("模型压缩包 sha256 校验失败")

        extracted_root = _extract_to_staging(zip_path, tmp_dir / "extract")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(extracted_root, model_dir)

    _validate_optimal_manifest(cfg, model_dir)

