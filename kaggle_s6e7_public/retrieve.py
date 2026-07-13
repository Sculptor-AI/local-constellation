from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

COMPETITION = "playground-series-s6e7"
ID = "id"
TARGET = "health_condition"
TOP_KERNELS = [
    "amanatar/s6e7-student-hearth-risk-lb-0-95112",
    "anhadmahajan06/s6e7-post-processing-ensemble-lb-0-95112",
    "makthanithin/s6e7-post-processing-ensemble-lb-0-95112",
]
OTHER_KERNELS = [
    "dalloliogm/ps6e7-automated-public-ensemble-v2",
]
PUBLIC_GITHUB = {
    "hook_lb_094967": (
        "https://raw.githubusercontent.com/Hook12aaa/kaggle-health-ps-s6e7/"
        "5f9a7dac59a8431482cf486645e773887e4dbd6f/submission.csv"
    ),
    "mrjohnson_final": (
        "https://raw.githubusercontent.com/mrjohnsonsea/playground-series-s6e7/"
        "d2ac3f0e767f14fc337c41d966a5a1fb16901604/data/submission/submission.csv"
    ),
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
    "Referer": "https://www.kaggle.com/",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_submission(path: Path, sample: pd.DataFrame) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"read failed: {type(exc).__name__}: {exc}"
    if list(df.columns) != list(sample.columns):
        return False, f"columns={df.columns.tolist()}"
    if len(df) != len(sample):
        return False, f"rows={len(df)} expected={len(sample)}"
    if not df[ID].equals(sample[ID]):
        return False, "id values/order mismatch"
    if not set(df[TARGET].dropna().unique()).issubset({"at-risk", "fit", "unhealthy"}):
        return False, "invalid labels"
    return True, "ok"


def collect_valid(root: Path, sample: pd.DataFrame, dest: Path, prefix: str) -> list[Path]:
    found: list[Path] = []
    candidates = sorted(set(root.rglob("*.csv")))
    for i, csv in enumerate(candidates):
        ok, reason = valid_submission(csv, sample)
        print(f"candidate {csv}: {reason}", flush=True)
        if ok:
            target = dest / f"submission_{prefix}_{i:02d}.csv"
            shutil.copy2(csv, target)
            found.append(target)
    return found


def try_direct(ref: str, work: Path, sample: pd.DataFrame, out: Path) -> tuple[list[Path], dict]:
    slug = ref.replace("/", "__")
    endpoint = f"https://www.kaggle.com/api/v1/kernels/output/{ref}"
    archive = work / f"direct_{slug}.bin"
    report: dict = {"endpoint": endpoint}
    try:
        with requests.get(endpoint, headers=HEADERS, timeout=(30, 300), stream=True, allow_redirects=True) as r:
            report.update({
                "status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "final_url": r.url,
            })
            with archive.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        report["bytes"] = archive.stat().st_size
        report["head"] = archive.read_bytes()[:200].decode("utf-8", errors="replace")
        extract = work / f"direct_{slug}"
        extract.mkdir(exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                report["members"] = zf.namelist()[:100]
                zf.extractall(extract)
        else:
            report["zip"] = False
        return collect_valid(extract, sample, out, f"direct_{slug}"), report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return [], report


def try_kagglehub(ref: str, work: Path, sample: pd.DataFrame, out: Path) -> tuple[list[Path], dict]:
    slug = ref.replace("/", "__")
    report: dict = {}
    try:
        import kagglehub

        path = Path(kagglehub.notebook_output_download(ref, force_download=True))
        report["path"] = str(path)
        report["exists"] = path.exists()
        if path.exists():
            local = work / f"kagglehub_{slug}"
            if local.exists():
                shutil.rmtree(local)
            if path.is_dir():
                shutil.copytree(path, local)
            else:
                local.mkdir()
                shutil.copy2(path, local / path.name)
            return collect_valid(local, sample, out, f"kagglehub_{slug}"), report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return [], report


def try_cli(ref: str, work: Path, sample: pd.DataFrame, out: Path) -> tuple[list[Path], dict]:
    slug = ref.replace("/", "__")
    target = work / f"cli_{slug}"
    target.mkdir(exist_ok=True)
    cmd = ["kaggle", "kernels", "output", ref, "-p", str(target)]
    report: dict = {"cmd": " ".join(cmd[:4])}
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
        report.update({
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        })
        return collect_valid(target, sample, out, f"cli_{slug}"), report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return [], report


def majority_vote(paths: list[Path], sample: pd.DataFrame, target: Path) -> dict:
    frames = [pd.read_csv(p) for p in paths]
    labels = []
    order = {"at-risk": 0, "fit": 1, "unhealthy": 2}
    for row in zip(*(df[TARGET].astype(str) for df in frames)):
        counts = Counter(row)
        labels.append(sorted(counts, key=lambda x: (-counts[x], order[x]))[0])
    result = sample.copy()
    result[TARGET] = labels
    result.to_csv(target, index=False)
    return {
        "members": [p.name for p in paths],
        "sha256": sha256(target),
        "distribution": result[TARGET].value_counts(normalize=True).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output / "work"
    work.mkdir(exist_ok=True)
    sample = pd.read_csv(args.sample)
    assert len(sample) == 295753

    report: dict = {"kernels": {}, "github": {}}
    top_valid: list[Path] = []
    all_valid: list[Path] = []

    for ref in TOP_KERNELS + OTHER_KERNELS:
        print(f"\n=== {ref} ===", flush=True)
        block: dict = {}
        valid_for_ref: list[Path] = []
        for name, fn in [("direct", try_direct), ("kagglehub", try_kagglehub), ("cli", try_cli)]:
            found, details = fn(ref, work, sample, args.output)
            block[name] = details
            valid_for_ref.extend(found)
            if found:
                break
        # de-duplicate byte-identical files.
        unique: dict[str, Path] = {}
        for p in valid_for_ref:
            unique.setdefault(sha256(p), p)
        valid_for_ref = list(unique.values())
        block["valid"] = [p.name for p in valid_for_ref]
        report["kernels"][ref] = block
        all_valid.extend(valid_for_ref)
        if ref in TOP_KERNELS:
            top_valid.extend(valid_for_ref)

    for name, url in PUBLIC_GITHUB.items():
        target = args.output / f"submission_{name}.csv"
        details: dict = {"url": url}
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
            details.update({"status": r.status_code, "bytes": len(r.content), "content_type": r.headers.get("content-type", "")})
            r.raise_for_status()
            target.write_bytes(r.content)
            ok, reason = valid_submission(target, sample)
            details["validation"] = reason
            if ok:
                details["sha256"] = sha256(target)
                details["distribution"] = pd.read_csv(target)[TARGET].value_counts(normalize=True).to_dict()
                all_valid.append(target)
            else:
                target.unlink(missing_ok=True)
        except Exception as exc:
            details["error"] = f"{type(exc).__name__}: {exc}"
        report["github"][name] = details

    # Deduplicate valid submissions by content hash.
    unique_all: dict[str, Path] = {}
    for p in all_valid:
        unique_all.setdefault(sha256(p), p)
    all_valid = list(unique_all.values())
    unique_top: dict[str, Path] = {}
    for p in top_valid:
        unique_top.setdefault(sha256(p), p)
    top_valid = list(unique_top.values())

    if top_valid:
        best = args.output / "submission_public_top_candidate.csv"
        shutil.copy2(top_valid[0], best)
        report["public_top_candidate"] = {
            "source": top_valid[0].name,
            "sha256": sha256(best),
            "distribution": pd.read_csv(best)[TARGET].value_counts(normalize=True).to_dict(),
        }
    if len(top_valid) >= 2:
        report["top_majority"] = majority_vote(top_valid, sample, args.output / "submission_public_top_majority.csv")
    if len(all_valid) >= 3:
        report["all_majority"] = majority_vote(all_valid, sample, args.output / "submission_public_all_majority.csv")

    report["valid_files"] = [p.name for p in all_valid]
    (args.output / "public_artifacts_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
