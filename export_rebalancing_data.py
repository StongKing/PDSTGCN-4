"""Export saved PDSTGCN predictions and validation-calibrated rebalancing data.

Run with the project's Python environment. No training and no changes to source
predictions/checkpoints. Error convention throughout: actual minus prediction.
See REBALANCING_DATA_GUIDE.md for column definitions and examples.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re

# Match the existing project's Windows runtime setup, before numerical imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "01a0a96b-0035-7f22-b360-8fc2b56e2ddf"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8g")


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def bounds(values, coverage=0.95):
    """Signed central interval, plus observed extrema; sample axis is axis 0."""
    tail = (1 - coverage) / 2
    return np.quantile(values, [0, tail, 0.5, 1-tail, 1], axis=0)


def audit_predictions(pred, target, fleet, name):
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"{name}: prediction and target shapes disagree")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError(f"{name}: nonfinite prediction/target")
    if pred.min() < -1e-6 or target.min() < -1e-6:
        raise ValueError(f"{name}: negative inventory")
    if np.max(np.abs(target.sum(axis=1)-fleet)) > .01:
        raise ValueError(f"{name}: labels do not conserve fleet")
    if np.max(np.abs(pred.sum(axis=1)-fleet)) > .01:
        raise ValueError(f"{name}: predictions do not conserve fleet; cannot use error identity")
    return {"max_prediction_fleet_error": float(np.abs(pred.sum(1)-fleet).max()),
            "max_error_identity_residual": float(np.abs((target-pred).sum(1)).max())}


def infer_validation(config, checkpoint, prepared, test_prediction, output):
    """Use exactly the same model/rounding as saved test output, never train."""
    import torch
    from model.ASTGCN_r import make_model
    from lib.utils import get_adjacency_matrix, SparseDynamicGraphStore

    d, t = config["Data"], config["Training"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    adj, _ = get_adjacency_matrix(str(ROOT / d["adj_filename"]), int(d["num_of_vertices"]))
    graph = SparseDynamicGraphStore.load(ROOT / d["dynamic_graph_filename"])
    net = make_model(device, int(t["nb_block"]), int(t["in_channels"]), int(t["K"]),
                     int(t["nb_chev_filter"]), int(t["nb_time_filter"]),
                     int(t["num_of_hours"]), adj, int(d["num_for_predict"]),
                     int(d["len_input"]), int(d["num_of_vertices"]), float(d["fleet_size"]))
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    net.eval()
    batch_size = int(t["batch_size"])
    integer = np.max(np.abs(test_prediction-np.rint(test_prediction))) < 1e-6
    with torch.no_grad():
        # Check a saved test batch against the checkpoint before calibration.
        probe = min(batch_size, len(test_prediction))
        probe_x = torch.as_tensor(prepared["test_x"][:probe], device=device)
        probe_g = graph.dense(prepared["test_graph_id"][:probe], device)
        actual = net(probe_x, probe_g, apply_rounding=integer).cpu().numpy()
        if not np.allclose(actual, test_prediction[:probe], atol=1e-4, rtol=0):
            raise ValueError("Checkpoint does not reproduce the saved test batch. "
                             "Select its original checkpoint with --checkpoint.")
        predictions = []
        for k in range(0, len(prepared["val_x"]), batch_size):
            x = torch.as_tensor(prepared["val_x"][k:k+batch_size], device=device)
            g = graph.dense(prepared["val_graph_id"][k:k+batch_size], device)
            predictions.append(net(x, g, apply_rounding=integer).cpu().numpy())
            if k % (10*batch_size) == 0:
                print(f"Validation inference: {k}/{len(prepared['val_x'])}", flush=True)
    pred = np.concatenate(predictions)
    np.savez_compressed(output, prediction=pred,
                        data_target_tensor=prepared["val_target"],
                        graph_id=prepared["val_graph_id"],
                        checkpoint_sha256=np.array(sha256(checkpoint)))
    return pred.astype(np.float64)


def station_summary(pred, target, mapping, minutes):
    error = target-pred
    result = []
    for h, duration in enumerate(minutes):
        table = mapping.copy()
        e = error[:, :, h]
        q = bounds(e)
        table["horizon_minutes"] = duration
        table["mae"] = np.abs(e).mean(0)
        table["rmse"] = np.sqrt(np.square(e).mean(0))
        table["bias_actual_minus_prediction"] = e.mean(0)
        for name, arr in zip(["min_error", "q025_error", "median_error", "q975_error", "max_error"], q):
            table[name] = arr
        result.append(table)
    return pd.concat(result, ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--horizon-minutes", type=int, default=60)
    p.add_argument("--sample-id", type=int, help="Zero-based test sample_id from scenario_index.csv")
    p.add_argument("--station-count", type=int, default=10)
    p.add_argument("--station-ids", help="Comma-separated ORIGINAL station IDs, not node indices")
    p.add_argument("--target-inventory", type=Path, help="CSV with station_id,target_inventory")
    p.add_argument("--skip-full-csv", action="store_true", help="Still export all arrays in NPZ")
    a = p.parse_args(argv)
    config = configparser.ConfigParser()
    config.read(ROOT / "configurations/DIVVY_astgcn.conf")
    d, t = config["Data"], config["Training"]
    exp = ROOT / "experiments" / d["dataset_name"] / (
        f"{t['model_name']}_h{t['num_of_hours']}d{t['num_of_days']}w{t['num_of_weeks']}"
        f"_channel{t['in_channels']}_{float(t['learning_rate']):.6e}")
    if a.prediction is None:
        match = re.search(r"^best_epoch=(\d+)$", (exp / "best_epoch.txt").read_text(), re.M)
        if not match:
            raise ValueError("best_epoch.txt has no best_epoch; specify --prediction")
        a.prediction = exp / f"output_epoch_{match[1]}_test.npz"
    a.prediction = a.prediction.resolve()
    epoch_match = re.search(r"output_epoch_(\d+)_test", a.prediction.stem)
    if a.checkpoint is None:
        if not epoch_match:
            raise ValueError("Specify --checkpoint for a nonstandard prediction filename")
        a.checkpoint = a.prediction.parent / f"epoch_{epoch_match[1]}.params"
    epoch = epoch_match[1] if epoch_match else "custom"
    out = (a.output_dir or DEFAULT_OUTPUT / f"rebalancing_epoch_{epoch}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    prepared_path = ROOT / "data/DIVVY/DIVVY_r1_d0_w0_astcgn.npz"
    prepared = np.load(prepared_path, allow_pickle=False)
    saved = np.load(a.prediction, allow_pickle=False)
    pred = saved["prediction"].astype(np.float64)
    target = saved["data_target_tensor"].astype(np.float64)
    if not np.array_equal(target, prepared["test_target"]):
        raise ValueError("Saved target/order does not match the current prepared test split")
    if "input" not in saved or not np.allclose(saved["input"], prepared["test_x"][:, :, :1, :], atol=1e-6):
        raise ValueError("Saved historical input/order does not match prepared test data")
    source = np.load(ROOT / d["graph_signal_matrix_filename"], allow_pickle=False)
    raw = source["data"][:, :, 0].astype(np.float64)
    times = pd.to_datetime(source["state_times_ns"])
    fleet = float(d["fleet_size"])
    audit = audit_predictions(pred, target, fleet, "test")
    s, n, horizon_count = pred.shape
    step_minutes = int((times[1]-times[0]).total_seconds()/60)
    minutes = list(range(step_minutes, (horizon_count+1)*step_minutes, step_minutes))
    if a.horizon_minutes not in minutes:
        raise ValueError(f"Choose horizon_minutes from {minutes}")
    hsel = minutes.index(a.horizon_minutes)
    mapping = pd.read_csv(ROOT / "data/DIVVY/node_mapping.csv", dtype={"node_id": str})
    mapping = mapping.sort_values("node_index").reset_index(drop=True)
    if not np.array_equal(mapping.node_index, np.arange(n)) or list(mapping.node_id) != list(source["node_ids"]):
        raise ValueError("Node mapping/order mismatch")
    if mapping.node_type.iloc[0] != "mesoscopic_in_transit":
        raise ValueError("Expected node 0 at position 0")
    timestamps = prepared["test_timestamp"].ravel().astype(int)
    gids = prepared["test_graph_id"].ravel().astype(int)
    sample_index = pd.read_csv(ROOT / "data/DIVVY/sample_index.csv").set_index("graph_id")
    matched = sample_index.loc[gids]
    if not np.array_equal(matched.target_start_index, timestamps) or not (matched["split"] == "test").all():
        raise ValueError("Sample timestamps / graph IDs mismatch")
    current = raw[timestamps-1]
    origins = times[timestamps-1]
    future_indices = timestamps[:, None]+np.arange(horizon_count)[None, :]
    if not np.array_equal(target, raw[future_indices].transpose(0, 2, 1)):
        raise ValueError("Targets do not match raw state timestamps")

    cache = out / "validation_predictions.npz"
    chash = sha256(a.checkpoint)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["checkpoint_sha256"]) != chash or not np.array_equal(z["data_target_tensor"], prepared["val_target"]):
            raise ValueError("Cached validation predictions belong to different weights/data; use a new output directory")
        vpred = z["prediction"].astype(np.float64)
    else:
        vpred = infer_validation(config, a.checkpoint, prepared, pred, cache)
    vy = prepared["val_target"].astype(np.float64)
    audit["validation"] = audit_predictions(vpred, vy, fleet, "validation")
    # Remove calibration windows with targets after the first test decision time.
    vts = prepared["val_timestamp"].ravel().astype(int)
    keep = vts+horizon_count-1 <= timestamps[0]-1
    if keep.sum() < 40:
        raise ValueError("Too few chronological calibration windows")
    ve = (vy-vpred)[keep]
    error = target-pred
    q = bounds(ve)
    n0q = q[:, 0, :]
    interval_rows = []
    for h, duration in enumerate(minutes):
        lo, hi = -n0q[3, h], -n0q[1, h]
        sumerr = error[:, 1:, h].sum(1)
        interval_rows.append({"horizon_minutes": duration, "calibration_samples": int(keep.sum()),
            "node0_min_error": n0q[0, h], "node0_q025_error": n0q[1, h],
            "node0_median_error": n0q[2, h], "node0_q975_error": n0q[3, h], "node0_max_error": n0q[4, h],
            "station_sum_min": -n0q[4, h], "station_sum_max": -n0q[0, h],
            "station_sum_q025": lo, "station_sum_q975": hi,
            "test_coverage_central95": float(((sumerr >= lo)&(sumerr <= hi)).mean()),
            "test_coverage_minmax": float(((sumerr >= -n0q[4,h])&(sumerr <= -n0q[0,h])).mean())})
    intervals = pd.DataFrame(interval_rows)
    csv(out / "robust_bounds_validation.csv", intervals)
    stats = station_summary(pred, target, mapping, minutes)
    csv(out / "station_metrics_test.csv", stats)
    local_bounds = mapping.iloc[np.tile(np.arange(n), horizon_count)].reset_index(drop=True)
    local_bounds["horizon_minutes"] = np.repeat(minutes, n)
    for key, value in zip(["error_min", "error_q025", "error_median", "error_q975", "error_max"], q):
        local_bounds[key] = value.T.ravel()
    csv(out / "station_bounds_validation.csv", local_bounds)

    scenario_rows = []
    for h, duration in enumerate(minutes):
        e = error[:, 1:, h]
        future = pred[:, 1:, h]
        change = future-current[:, 1:]
        table = pd.DataFrame({"sample_id": np.arange(s), "graph_id": gids,
            "forecast_origin": origins, "target_time": times[future_indices[:, h]],
            "horizon_minutes": duration, "predicted_change_abs_sum": np.abs(change).sum(1),
            "predicted_increase_total": np.maximum(change, 0).sum(1),
            "predicted_decrease_total": np.maximum(-change, 0).sum(1),
            "predicted_empty_station_count": (future <= 0).sum(1),
            "predicted_low_station_count_le2": (future <= 2).sum(1),
            "current_node0": current[:, 0], "predicted_node0": pred[:, 0, h],
            "actual_node0": target[:, 0, h], "node0_error": error[:, 0, h],
            "station_error_sum": e.sum(1), "station_mae": np.abs(e).mean(1),
            "station_rmse": np.sqrt(np.square(e).mean(1))})
        scenario_rows.append(table)
    scenarios = pd.concat(scenario_rows).sort_values(["sample_id", "horizon_minutes"]).reset_index(drop=True)
    csv(out / "scenario_index.csv", scenarios)
    # All original nodes retained. Arrays are [test sample, node, horizon].
    np.savez_compressed(out / "all_predictions.npz", prediction=pred, actual=target,
                        error_actual_minus_prediction=error, current_inventory=current,
                        node_ids=source["node_ids"], sample_id=np.arange(s), graph_id=gids,
                        origin_time_ns=times[timestamps-1].asi8,
                        target_time_ns=source["state_times_ns"][future_indices], horizon_minutes=minutes)
    np.savez_compressed(out / "calibration_errors.npz", error_actual_minus_prediction=ve,
                        node_ids=source["node_ids"], horizon_minutes=minutes,
                        graph_id=prepared["val_graph_id"].ravel()[keep],
                        target_start_time_ns=source["state_times_ns"][vts[keep]])
    csv(out / "node_mapping.csv", mapping)

    if not a.skip_full_csv:
        for h, duration in enumerate(minutes):
            # One file per horizon remains below Excel's 1,048,576-row limit.
            frame = mapping.iloc[np.tile(np.arange(n), s)].reset_index(drop=True)
            frame.insert(0, "sample_id", np.repeat(np.arange(s), n))
            frame.insert(1, "graph_id", np.repeat(gids, n))
            frame.insert(2, "forecast_origin", np.repeat(origins, n))
            frame.insert(3, "target_time", np.repeat(times[future_indices[:, h]], n))
            frame["horizon_minutes"] = duration
            frame["current_inventory"] = current.ravel()
            frame["predicted_inventory"] = pred[:, :, h].ravel()
            frame["actual_inventory"] = target[:, :, h].ravel()
            frame["error_actual_minus_prediction"] = error[:, :, h].ravel()
            frame["predicted_change_from_current"] = (pred[:, :, h]-current).ravel()
            csv(out / f"all_nodes_{duration:02d}min.csv.gz", frame)
            print(f"Exported {duration} min: {len(frame):,} rows", flush=True)

    if a.sample_id is None:
        # Fixed clock-time rule; does not inspect future actuals or errors.
        candidates = np.flatnonzero((origins.hour == 8)&(origins.minute == 0))
        sample = int(candidates[0]) if len(candidates) else 0
    else:
        sample = a.sample_id
    if not 0 <= sample < s:
        raise ValueError(f"sample_id must be in [0,{s-1}]")
    if a.station_ids:
        requested = [v.strip() for v in a.station_ids.split(",")]
        lookup = dict(zip(mapping.node_id, mapping.node_index))
        if len(set(requested)) != len(requested) or any(v not in lookup or v == "0" for v in requested):
            raise ValueError("station_ids must be unique original real-station IDs")
        selected = np.array([lookup[v] for v in requested], dtype=int)
        rule = "User-specified original station IDs"
    else:
        if not 1 <= a.station_count <= n-1:
            raise ValueError("station_count must be between 1 and real station count")
        changes = np.abs(pred[sample, 1:, hsel]-current[sample, 1:])
        selected = np.argsort(-changes, kind="stable")[:a.station_count]+1
        rule = "Largest absolute predicted changes from current stock; no test actual/error selection"
    other = np.setdiff1d(np.arange(1, n), selected)
    case = mapping.iloc[1:].copy()
    case.insert(0, "selected", np.isin(case.node_index, selected).astype(int))
    case["current_inventory"] = current[sample, 1:]
    case["predicted_inventory"] = pred[sample, 1:, hsel]
    case["actual_inventory"] = target[sample, 1:, hsel]
    case["error_actual_minus_prediction"] = error[sample, 1:, hsel]
    case["predicted_change_from_current"] = pred[sample, 1:, hsel]-current[sample, 1:]
    case["error_lower95"] = q[1, 1:, hsel]
    case["error_upper95"] = q[3, 1:, hsel]
    case["inventory_lower95"] = np.maximum(0, case.predicted_inventory+case.error_lower95)
    case["inventory_upper95"] = np.maximum(0, case.predicted_inventory+case.error_upper95)
    case["target_inventory"] = np.nan
    if a.target_inventory:
        targets = pd.read_csv(a.target_inventory, dtype={"station_id": str})
        if not {"station_id", "target_inventory"} <= set(targets):
            raise ValueError("Target CSV needs station_id,target_inventory")
        if targets.station_id.duplicated().any():
            raise ValueError("Duplicate target station IDs")
        vals = pd.to_numeric(targets.target_inventory, errors="raise")
        if vals.dropna().lt(0).any() or np.isinf(vals.dropna()).any():
            raise ValueError("Target stocks must be finite and nonnegative")
        if not set(targets.station_id) <= set(mapping.node_id.iloc[1:]):
            raise ValueError("Unknown station_id in target inventory")
        case["target_inventory"] = case.node_id.map(pd.Series(vals.values, index=targets.station_id))
    case["nominal_surplus"] = case.predicted_inventory-case.target_inventory
    case["required_addition"] = case.target_inventory-case.predicted_inventory
    case = case.sort_values(["selected", "node_index"], ascending=[False, True]).reset_index(drop=True)
    case_dir = out / f"case_s{sample:04d}_h{a.horizon_minutes:02d}"
    case_dir.mkdir(exist_ok=True)
    csv(case_dir / "all_stations.csv", case)
    csv(case_dir / "selected_stations.csv", case[case.selected == 1])
    csv(case_dir / "target_inventory_template.csv", pd.DataFrame({"station_id": case.node_id, "target_inventory": case.target_inventory}))
    # Preserve paired residual vectors, including the unselected stations.
    selected_e = ve[:, selected, hsel].sum(1)
    other_e = ve[:, other, hsel].sum(1)
    block = pd.DataFrame({"node0_error": ve[:, 0, hsel], "selected_station_error_sum": selected_e,
                          "other_station_error_sum": other_e})
    block["identity_residual"] = block.sum(axis=1)
    csv(case_dir / "paired_calibration_errors.csv", block)
    sq, oq = bounds(selected_e), bounds(other_e)
    hbounds = intervals.iloc[hsel].to_dict()
    meta = {"sample_id": sample, "graph_id": int(gids[sample]), "horizon_minutes": a.horizon_minutes,
            "forecast_origin": str(origins[sample]), "target_time": str(times[future_indices[sample, hsel]]),
            "station_ids": mapping.node_id.iloc[selected].tolist(), "selection_rule": rule,
            "selected_sum_empirical95": [float(sq[1]), float(sq[3])],
            "other_sum_empirical95": [float(oq[1]), float(oq[3])],
            "full_system_bounds": hbounds,
            "constraint": "sum(selected errors) + sum(other station errors) + node0_error = 0",
            "error_definition": "actual - prediction",
            "selected_interval_note": "Empirically calibrated on the fixed subset; not obtained by using the full-system bound unchanged.",
            "target_inventory_note": "User input. Predicted change from current inventory is not rebalancing demand."}
    save_json(case_dir / "case_parameters.json", meta)
    # Distances are straight-line kilometres, not road travel costs; no depot invented.
    selected_map = mapping.iloc[selected]
    lat = np.radians(selected_map.latitude.to_numpy(float))
    lon = np.radians(selected_map.longitude.to_numpy(float))
    hav = np.sin((lat[:,None]-lat[None,:])/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    distances = 6371.0*2*np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    dist_frame = pd.DataFrame(distances, columns=selected_map.node_id)
    dist_frame.insert(0, "station_id", selected_map.node_id.to_numpy())
    csv(case_dir / "straight_line_distance_km.csv", dist_frame)
    audit["calibration_samples_used"] = int(keep.sum())
    audit["calibration_samples_removed_at_boundary"] = int((~keep).sum())
    audit["test_samples"] = s
    audit["station_count"] = n-1
    audit["prediction_sha256"] = sha256(a.prediction)
    audit["checkpoint_sha256"] = chash
    audit["source_prediction"] = str(a.prediction)
    audit["source_checkpoint"] = str(a.checkpoint.resolve())
    audit["error_definition"] = "actual - prediction"
    audit["calibration_method"] = "Validation empirical 2.5%/97.5% quantiles, NumPy linear interpolation; no test labels used to fit bounds. Validation was also used for checkpoint selection."
    audit["data_time_alignment_note"] = "Existing dynamic graphs include the first target interval (previously audited); this export preserves current predictions for exploratory rebalancing tests."
    audit["case_directory"] = str(case_dir)
    audit["test_station_mae"] = float(np.abs(error[:,1:]).mean())
    audit["test_station_rmse"] = float(np.sqrt(np.square(error[:,1:]).mean()))
    audit["test_node0_mae"] = float(np.abs(error[:,0]).mean())
    save_json(out / "manifest.json", audit)
    workbook_data = {"manifest": audit, "case": meta, "case_rows": records(case),
        "bounds": records(intervals), "scenario_index": records(scenarios),
        "station_summary": records(stats[stats.horizon_minutes == a.horizon_minutes]),
        "selected_calibration_errors": records(block),
        "metric_by_horizon": [{"horizon_minutes": duration,
            "station_mae": float(np.abs(error[:,1:,h]).mean()),
            "station_rmse": float(np.sqrt(np.square(error[:,1:,h]).mean())),
            "node0_mae": float(np.abs(error[:,0,h]).mean()),
            "persistence_mae": float(np.abs(target[:,1:,h]-current[:,1:]).mean())}
            for h, duration in enumerate(minutes)]}
    save_json(out / "workbook_data.json", workbook_data)
    print(json.dumps({"output": str(out), "case": meta, "audit": audit}, ensure_ascii=False, indent=2), flush=True)
    return out


if __name__ == "__main__":
    main()
