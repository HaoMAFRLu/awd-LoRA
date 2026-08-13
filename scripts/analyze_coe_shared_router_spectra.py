#!/usr/bin/env python3
"""Download only CoE router tensors and compare their singular spectra."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import struct
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "moe_router_analysis" / "chain_of_experts"

MODELS = (
    {
        "key": "no_shared",
        "label": "0 shared (64 routed)",
        "repo": "chain-of-experts/64ept-4tpk-2itr-noSharedExp-metamathqa-2k",
        "revision": "c5f6fb0cc3d1a1792112fcefa554203c45ae5592",
        "expected_router_rows": 64,
    },
    {
        "key": "one_shared",
        "label": "1 shared (63 routed)",
        "repo": "chain-of-experts/64ept-4tpk-2itr-1SharedExp-metamathqa-2k",
        "revision": "de7ae215a08704b301d3adb37c31b1944d4c83a3",
        "expected_router_rows": 63,
    },
)

ROUTER_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\."
    r"(?P<iteration>\d+)\.weight$"
)

DTYPE_TO_NUMPY = {
    "F32": "<f4",
    "F16": "<f2",
}


def _read_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip()
    return None


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "coe-router-spectrum-analysis/1.0"})
    token = _read_hf_token()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def _resolve_url(model: dict[str, Any]) -> str:
    return (
        f"https://huggingface.co/{model['repo']}/resolve/"
        f"{model['revision']}/model.safetensors"
    )


def _get_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    *,
    max_attempts: int = 6,
) -> bytes:
    expected = end - start + 1
    for attempt in range(max_attempts):
        response = session.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            allow_redirects=True,
            stream=True,
            timeout=(20, 90),
        )
        if response.status_code == 206:
            data = response.content
            response.close()
            if len(data) != expected:
                raise RuntimeError(
                    f"short range response: requested {expected} bytes, "
                    f"received {len(data)}"
                )
            return data

        status = response.status_code
        message = response.text[:500]
        response.close()
        if status != 429 or attempt + 1 == max_attempts:
            raise RuntimeError(
                f"range request failed with HTTP {status}: {message}"
            )
        time.sleep(min(5 * (attempt + 1), 30))

    raise AssertionError("unreachable")


def _read_safetensors_header(
    session: requests.Session, model: dict[str, Any]
) -> tuple[int, dict[str, Any], int]:
    url = _resolve_url(model)
    prefix = _get_range(session, url, 0, 7)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size <= 0 or header_size > 100_000_000:
        raise RuntimeError(f"invalid safetensors header size: {header_size}")
    raw_header = _get_range(session, url, 8, 8 + header_size - 1)
    return header_size, json.loads(raw_header.decode("utf-8")), 8 + header_size


def _decode_tensor(raw: bytes, info: dict[str, Any]) -> np.ndarray:
    dtype = info["dtype"]
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2")
        values = (words.astype(np.uint32) << 16).view(np.float32)
    elif dtype in DTYPE_TO_NUMPY:
        values = np.frombuffer(raw, dtype=DTYPE_TO_NUMPY[dtype]).astype(
            np.float32, copy=False
        )
    else:
        raise ValueError(f"unsupported router dtype: {dtype}")
    return values.reshape(info["shape"]).copy()


def _extract_routers(
    session: requests.Session, model: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    header_size, header, data_start = _read_safetensors_header(session, model)
    router_entries = []
    for name, info in header.items():
        match = ROUTER_PATTERN.fullmatch(name)
        if match:
            router_entries.append(
                (
                    int(match.group("layer")),
                    int(match.group("iteration")),
                    name,
                    info,
                )
            )
    router_entries.sort()
    if len(router_entries) != 8:
        raise RuntimeError(
            f"expected 8 routers in {model['repo']}, found {len(router_entries)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    manifest_entries = []
    total_tensor_bytes = 0
    url = _resolve_url(model)
    for layer, iteration, name, info in router_entries:
        begin, finish = info["data_offsets"]
        raw = _get_range(
            session,
            url,
            data_start + begin,
            data_start + finish - 1,
        )
        weight = _decode_tensor(raw, info)
        expected_shape = (model["expected_router_rows"], 1024)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"unexpected router shape for {name}: {weight.shape}, "
                f"expected {expected_shape}"
            )
        tensors[name] = torch.from_numpy(weight)
        total_tensor_bytes += len(raw)
        manifest_entries.append(
            {
                "name": name,
                "layer": layer,
                "iteration": iteration,
                "dtype": info["dtype"],
                "shape": list(weight.shape),
                "source_data_offsets": [begin, finish],
            }
        )

    manifest = {
        "source_repo": model["repo"],
        "source_revision": model["revision"],
        "source_file": "model.safetensors",
        "source_header_size_bytes": header_size,
        "router_tensor_bytes_downloaded": total_tensor_bytes,
        "router_tensors": manifest_entries,
    }
    return tensors, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _energy_rank(singular_values: np.ndarray, threshold: float) -> int:
    energy = np.square(singular_values, dtype=np.float64)
    cumulative = np.cumsum(energy) / energy.sum()
    return int(np.searchsorted(cumulative, threshold) + 1)


def _spectrum_metrics(weight: np.ndarray) -> dict[str, Any]:
    weight64 = weight.astype(np.float64)
    singular = np.linalg.svd(weight64, compute_uv=False)
    centered = weight64 - weight64.mean(axis=0, keepdims=True)
    centered_singular = np.linalg.svd(centered, compute_uv=False)
    # Centering makes the rows exactly linearly dependent; discard the final
    # numerical round-off singular value.
    centered_singular = centered_singular[: weight.shape[0] - 1]

    def summarize(values: np.ndarray) -> dict[str, Any]:
        energy = np.square(values, dtype=np.float64)
        probabilities = energy / energy.sum()
        nonzero_probabilities = probabilities[probabilities > 0]
        effective_rank = float(
            np.exp(
                -np.sum(
                    nonzero_probabilities * np.log(nonzero_probabilities)
                )
            )
        )
        return {
            "singular_values": values.tolist(),
            "normalized_singular_values": (values / values[0]).tolist(),
            "spectral_energy_fractions": probabilities.tolist(),
            "cumulative_spectral_energy": np.cumsum(probabilities).tolist(),
            "effective_rank": effective_rank,
            "effective_rank_ratio": effective_rank / len(values),
            "stable_rank": float(energy.sum() / energy[0]),
            "stable_rank_ratio": float(energy.sum() / energy[0] / len(values)),
            "rank_50_energy": _energy_rank(values, 0.50),
            "rank_90_energy": _energy_rank(values, 0.90),
            "rank_99_energy": _energy_rank(values, 0.99),
            "top1_energy_fraction": float(probabilities[0]),
            "top5_energy_fraction": float(probabilities[:5].sum()),
            "smallest_to_largest_ratio": float(values[-1] / values[0]),
        }

    return {
        "raw": summarize(singular),
        "row_centered": summarize(centered_singular),
        "common_row_energy_fraction": float(
            1.0 - np.square(centered).sum() / np.square(weight64).sum()
        ),
    }


def _analyze(
    all_tensors: dict[str, dict[str, torch.Tensor]]
) -> dict[str, Any]:
    results: dict[str, Any] = {"models": {}}
    for model in MODELS:
        model_results = []
        for name, tensor in sorted(all_tensors[model["key"]].items()):
            match = ROUTER_PATTERN.fullmatch(name)
            assert match is not None
            metrics = _spectrum_metrics(tensor.numpy())
            model_results.append(
                {
                    "tensor_name": name,
                    "layer": int(match.group("layer")),
                    "iteration": int(match.group("iteration")),
                    "shape": list(tensor.shape),
                    **metrics,
                }
            )
        model_results.sort(key=lambda item: (item["layer"], item["iteration"]))
        results["models"][model["key"]] = {
            "label": model["label"],
            "repo": model["repo"],
            "routers": model_results,
        }
    return results


def _router_lookup(
    results: dict[str, Any], model_key: str, layer: int, iteration: int
) -> dict[str, Any]:
    routers = results["models"][model_key]["routers"]
    return next(
        router
        for router in routers
        if router["layer"] == layer and router["iteration"] == iteration
    )


def _plot_grid(
    results: dict[str, Any],
    metric_path: tuple[str, str],
    ylabel: str,
    filename: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.5), sharex=False)
    colors = {"no_shared": "#2474b5", "one_shared": "#d95f02"}

    for layer in range(4):
        for iteration in range(2):
            axis = axes[layer, iteration]
            for model in MODELS:
                router = _router_lookup(
                    results, model["key"], layer, iteration
                )
                values = router[metric_path[0]][metric_path[1]]
                x = np.arange(1, len(values) + 1)
                axis.plot(
                    x,
                    values,
                    linewidth=1.8,
                    color=colors[model["key"]],
                    label=model["label"],
                )
            axis.set_title(f"Block {layer}, routing iteration {iteration}")
            axis.set_xlabel("Singular-value index")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if ylim is not None:
                axis.set_ylim(*ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def _write_summary_csv(results: dict[str, Any]) -> Path:
    analysis_dir = OUTPUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "router_spectrum_summary.csv"
    fields = (
        "model",
        "layer",
        "iteration",
        "rows",
        "columns",
        "raw_effective_rank",
        "raw_effective_rank_ratio",
        "raw_stable_rank",
        "raw_rank_90_energy",
        "raw_rank_99_energy",
        "raw_top1_energy_fraction",
        "common_row_energy_fraction",
        "centered_effective_rank",
        "centered_effective_rank_ratio",
        "centered_stable_rank",
        "centered_rank_90_energy",
        "centered_rank_99_energy",
        "centered_top1_energy_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            for router in results["models"][model["key"]]["routers"]:
                raw = router["raw"]
                centered = router["row_centered"]
                writer.writerow(
                    {
                        "model": model["key"],
                        "layer": router["layer"],
                        "iteration": router["iteration"],
                        "rows": router["shape"][0],
                        "columns": router["shape"][1],
                        "raw_effective_rank": raw["effective_rank"],
                        "raw_effective_rank_ratio": raw["effective_rank_ratio"],
                        "raw_stable_rank": raw["stable_rank"],
                        "raw_rank_90_energy": raw["rank_90_energy"],
                        "raw_rank_99_energy": raw["rank_99_energy"],
                        "raw_top1_energy_fraction": raw["top1_energy_fraction"],
                        "common_row_energy_fraction": router[
                            "common_row_energy_fraction"
                        ],
                        "centered_effective_rank": centered["effective_rank"],
                        "centered_effective_rank_ratio": centered[
                            "effective_rank_ratio"
                        ],
                        "centered_stable_rank": centered["stable_rank"],
                        "centered_rank_90_energy": centered["rank_90_energy"],
                        "centered_rank_99_energy": centered["rank_99_energy"],
                        "centered_top1_energy_fraction": centered[
                            "top1_energy_fraction"
                        ],
                    }
                )
    return path


def _aggregate(results: dict[str, Any]) -> dict[str, Any]:
    aggregate = {}
    for model in MODELS:
        routers = results["models"][model["key"]]["routers"]
        model_aggregate = {}
        for spectrum_kind in ("raw", "row_centered"):
            model_aggregate[spectrum_kind] = {}
            for metric in (
                "effective_rank",
                "effective_rank_ratio",
                "stable_rank",
                "stable_rank_ratio",
                "rank_90_energy",
                "rank_99_energy",
                "top1_energy_fraction",
            ):
                values = np.asarray(
                    [router[spectrum_kind][metric] for router in routers],
                    dtype=np.float64,
                )
                model_aggregate[spectrum_kind][metric] = {
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
        aggregate[model["key"]] = model_aggregate
        common_energy = np.asarray(
            [router["common_row_energy_fraction"] for router in routers],
            dtype=np.float64,
        )
        aggregate[model["key"]]["common_row_energy_fraction"] = {
            "mean": float(common_energy.mean()),
            "min": float(common_energy.min()),
            "max": float(common_energy.max()),
        }
    return aggregate


def main() -> None:
    session = _session()
    all_tensors: dict[str, dict[str, torch.Tensor]] = {}
    manifests = {}

    for model in MODELS:
        model_dir = OUTPUT_ROOT / model["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        router_path = model_dir / "router_only.safetensors"
        manifest_path = model_dir / "manifest.json"
        cached_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if router_path.is_file() and manifest_path.is_file()
            else None
        )
        if (
            cached_manifest is not None
            and cached_manifest.get("source_revision") == model["revision"]
            and cached_manifest.get("router_file_sha256") == _sha256(router_path)
        ):
            print(f"Reusing routers from {router_path.relative_to(ROOT)}...", flush=True)
            tensors = load_file(str(router_path))
            manifest = cached_manifest
        else:
            print(f"Extracting routers from {model['repo']}...", flush=True)
            tensors, manifest = _extract_routers(session, model)
            save_file(
                tensors,
                str(router_path),
                metadata={
                    "source_repo": model["repo"],
                    "source_revision": model["revision"],
                },
            )
            manifest["router_file"] = str(router_path.relative_to(ROOT))
            manifest["router_file_size_bytes"] = router_path.stat().st_size
            manifest["router_file_sha256"] = _sha256(router_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        all_tensors[model["key"]] = tensors
        manifests[model["key"]] = manifest

    results = _analyze(all_tensors)
    results["aggregate"] = _aggregate(results)
    results["manifests"] = manifests

    analysis_dir = OUTPUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = analysis_dir / "router_spectrum_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = _write_summary_csv(results)

    _plot_grid(
        results,
        ("raw", "singular_values"),
        "Singular value",
        "router_raw_singular_values",
    )
    _plot_grid(
        results,
        ("raw", "normalized_singular_values"),
        "Normalized singular value",
        "router_normalized_singular_values",
        ylim=(0.0, 1.03),
    )
    _plot_grid(
        results,
        ("row_centered", "normalized_singular_values"),
        "Normalized singular value (row-centered)",
        "router_centered_normalized_singular_values",
        ylim=(0.0, 1.03),
    )
    _plot_grid(
        results,
        ("row_centered", "cumulative_spectral_energy"),
        "Cumulative spectral energy (row-centered)",
        "router_centered_cumulative_energy",
        ylim=(0.0, 1.03),
    )

    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "summary_csv": str(csv_path.relative_to(ROOT)),
                "aggregate": results["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
