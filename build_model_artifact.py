from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json_obj(obj: dict) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="构建固定模型发布制品（含最优校验 manifest）")
    parser.add_argument("--model-dir", default="models/fixed_15m_stacking", help="本地模型目录")
    parser.add_argument("--opt-json", default="outputs/opt_15m_details.json", help="优化结果 JSON")
    parser.add_argument("--out-zip", default="artifacts/fixed_15m_stacking_release.zip", help="输出 zip 路径")
    parser.add_argument("--model-name", default="ensemble_stacking", help="模型名称标识")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_dir = root / args.model_dir
    opt_path = root / args.opt_json
    out_zip = root / args.out_zip

    required = ("logistic.joblib", "lightgbm.joblib", "catboost.joblib", "stacking_meta.joblib", "meta.json")
    miss = [f for f in required if not (model_dir / f).exists()]
    if miss:
        raise RuntimeError(f"模型目录不完整，缺少文件: {miss}")
    if not opt_path.exists():
        raise RuntimeError(f"找不到优化结果: {opt_path}")

    opt = json.loads(opt_path.read_text(encoding="utf-8"))
    best_subset = list(opt.get("best_stacking", {}).get("subset", []))
    best_meta_params = dict(opt.get("best_stacking", {}).get("meta_params", {}))
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    model_subset = list(meta.get("stack_subset", []))
    is_best = bool(best_subset) and model_subset == best_subset
    if not is_best:
        raise RuntimeError(
            f"模型 stack_subset 与 best_stacking 不一致，拒绝打包。model={model_subset}, best={best_subset}"
        )

    manifest = {
        "model_name": str(args.model_name),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_best_from_opt": True,
        "opt_sha256": _sha256_file(opt_path),
        "best_stacking_subset": best_subset,
        "best_stacking_meta_params_hash": _sha256_json_obj(best_meta_params),
        "model_stack_subset": model_subset,
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model_artifact_build_") as td:
        staging = Path(td) / "fixed_15m_stacking"
        shutil.copytree(model_dir, staging)
        (staging / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
            for p in sorted(staging.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(staging).as_posix())

    zip_sha = _sha256_file(out_zip)
    print(json.dumps({"ok": True, "out_zip": str(out_zip), "zip_sha256": zip_sha, "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

