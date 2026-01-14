"""This script is used for ablation studies on hyperparameters
"""
import sys, os
from collections import defaultdict
from math import isfinite

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

def _to_float(x):
    try:
        v = float(x)
        if not isfinite(v):
            return None
        return v
    except Exception:
        return None

def _to_float_or_none(x):
    try:
        v = float(x)
        if not isfinite(v):
            return None
        return v
    except Exception:
        return None
    
def _safe_float(x):
    try:
        v = float(x)
        if not isfinite(v):
            return None
        return v
    except Exception:
        return None

def find_rho_sweeps_same_dalpha_dbeta(file_list, min_rhos=2, dedup_strategy="best_ppl_slr"):
    """
    dedup_strategy:
      - "best_ppl_slr": if multiple rows share same (dalpha, dbeta, rho), pick the one with smallest ppl_SLR
      - "first": pick the first occurrence
    Returns: dict keyed by (dalpha, dbeta) -> list of rows (one per rho), sorted by rho
    Each row contains: rho, ppl_X, ppl_SLR, nr_SLR, file
    """
    # group by (dalpha, dbeta) -> list of rows
    groups = defaultdict(list)

    # normalize and collect
    for row in file_list:
        rho = _to_float(row.get("rho"))
        dalpha = _to_float(row.get("dalpha"))
        dbeta = _to_float(row.get("dbeta"))
        if rho is None or dalpha is None or dbeta is None:
            continue

        groups[(dalpha, dbeta)].append({
            "rho": rho,
            "dalpha": dalpha,
            "dbeta": dbeta,
            "ppl_X": row.get("ppl_X"),
            "ppl_SLR": row.get("ppl_SLR"),
            "nr_SLR": row.get("nr_SLR"),
            "file": row.get("file"),
        })

    results = {}

    for key, rows in groups.items():
        # distinct rho count
        rhos = sorted({r["rho"] for r in rows})
        if len(rhos) < min_rhos:
            continue

        # deduplicate per rho
        by_rho = defaultdict(list)
        for r in rows:
            by_rho[r["rho"]].append(r)

        chosen = []
        for rho_val, rs in by_rho.items():
            if dedup_strategy == "first":
                chosen.append(rs[0])
            else:
                # pick smallest ppl_SLR if numeric; otherwise fall back to first
                def ppl_slr_key(x):
                    v = _to_float_or_none(x.get("ppl_SLR"))
                    return v if v is not None else float("inf")
                rs_sorted = sorted(rs, key=ppl_slr_key)
                chosen.append(rs_sorted[0])

        # sort by rho
        chosen_sorted = sorted(chosen, key=lambda r: r["rho"])
        results[key] = chosen_sorted

    return results

def print_rho_sweeps(results, max_groups=None):
    """
    Pretty print results keyed by (dalpha, dbeta).
    """
    keys = sorted(results.keys())  # deterministic order
    if max_groups is not None:
        keys = keys[:max_groups]

    for (dalpha, dbeta) in keys:
        rows = results[(dalpha, dbeta)]
        print("=" * 72)
        print(f"dalpha = {dalpha}, dbeta = {dbeta} | #rho = {len(rows)}")
        print("rho\tppl_X\tppl_L+S\t nr_L+S\tfile")
        for r in rows:
            print(f"{r['rho']}\t{r.get('ppl_X')}\t{r.get('ppl_SLR')}\t{r.get('nr_SLR')}\t{r.get('file')}")
    if not keys:
        print("No (dalpha, dbeta) pairs found with >= 2 distinct rho values.")

def find_longest_sequences(file_list):
    # 1) 固定 (rho, dalpha) -> dbeta 序列最长
    group_dbeta = defaultdict(list)  # key=(rho, dalpha) -> rows
    # 2) 固定 (rho, dbeta) -> dalpha 序列最长
    group_dalpha = defaultdict(list) # key=(rho, dbeta) -> rows

    # 统一做 float 化，避免字符串/科学计数/None 导致分组错乱
    normalized_rows = []
    for row in file_list:
        rho = _safe_float(row.get("rho"))
        dalpha = _safe_float(row.get("dalpha"))
        dbeta = _safe_float(row.get("dbeta"))
        if rho is None or dalpha is None or dbeta is None:
            continue

        ppl_X   = row.get("ppl_X")
        ppl_SLR = row.get("ppl_SLR")
        nr_SLR  = row.get("nr_SLR")
        f       = row.get("file")

        normalized_rows.append({
            "file": f,
            "rho": rho,
            "dalpha": dalpha,
            "dbeta": dbeta,
            "ppl_X": ppl_X,
            "ppl_SLR": ppl_SLR,
            "nr_SLR": nr_SLR,
        })

    for row in normalized_rows:
        group_dbeta[(row["rho"], row["dalpha"])].append(row)
        group_dalpha[(row["rho"], row["dbeta"])].append(row)

    def _best_group(group_dict, vary_key):
        """
        group_dict: key -> list[rows]
        vary_key: 'dbeta' or 'dalpha'
        choose group with max number of distinct vary_key values
        tie-breakers:
          (1) more total rows
          (2) smaller rho (arbitrary but deterministic)
          (3) smaller other key (dalpha or dbeta)
        """
        best_key = None
        best_rows = None
        best_distinct = -1
        best_total = -1

        for key, rows in group_dict.items():
            distinct_vals = sorted({r[vary_key] for r in rows})
            n_distinct = len(distinct_vals)
            n_total = len(rows)

            if (n_distinct > best_distinct or
                (n_distinct == best_distinct and n_total > best_total) or
                (n_distinct == best_distinct and n_total == best_total and (best_key is None or key < best_key))):
                best_key = key
                best_rows = rows
                best_distinct = n_distinct
                best_total = n_total

        if best_key is None:
            return None, [], 0

        # 对同一个 vary_key 可能有重复实验点（同样的 dbeta/dalpha 多个 file）
        # 这里按 vary_key 聚合：若重复，取 ppl_SLR 更小的那条（你也可以改成平均）
        by_val = defaultdict(list)
        for r in best_rows:
            by_val[r[vary_key]].append(r)

        chosen = []
        for v, rs in by_val.items():
            # 以 ppl_SLR 作为优先选择的指标（越小越好）；如果不可比就取第一条
            def _ppl_slr_or_inf(x):
                pv = x.get("ppl_SLR")
                try:
                    return float(pv)
                except Exception:
                    return float("inf")
            rs_sorted = sorted(rs, key=_ppl_slr_or_inf)
            chosen.append(rs_sorted[0])

        chosen_sorted = sorted(chosen, key=lambda r: r[vary_key])
        return best_key, chosen_sorted, len(chosen_sorted)

    # A) 固定 (rho, dalpha) 看 dbeta
    best_key_dbeta, rows_dbeta, n_dbeta = _best_group(group_dbeta, "dbeta")
    # B) 固定 (rho, dbeta) 看 dalpha
    best_key_dalpha, rows_dalpha, n_dalpha = _best_group(group_dalpha, "dalpha")

    return (best_key_dbeta, rows_dbeta, n_dbeta), (best_key_dalpha, rows_dalpha, n_dalpha)

def pretty_print_dbeta_ablation(best_key_dbeta, rows_dbeta):
    rho, dalpha = best_key_dbeta
    print("=== Longest dbeta sequence (fix rho, dalpha) ===")
    print(f"Fixed rho   = {rho}")
    print(f"Fixed dalpha= {dalpha}")
    print(f"Num dbeta   = {len(rows_dbeta)}")
    print("dbeta\tppl_X\tppl_SLR\tnr_SLR\tfile")
    for r in rows_dbeta:
        print(f"{r['dbeta']}\t{r.get('ppl_X')}\t{r.get('ppl_SLR')}\t{r.get('nr_SLR')}\t{r.get('file')}")

def pretty_print_dalpha_ablation(best_key_dalpha, rows_dalpha):
    rho, dbeta = best_key_dalpha
    print("=== Longest dalpha sequence (fix rho, dbeta) ===")
    print(f"Fixed rho  = {rho}")
    print(f"Fixed dbeta= {dbeta}")
    print(f"Num dalpha = {len(rows_dalpha)}")
    print("dalpha\tppl_X\tppl_SLR\tnr_SLR\tfile")
    for r in rows_dalpha:
        print(f"{r['dalpha']}\t{r.get('ppl_X')}\t{r.get('ppl_SLR')}\t{r.get('nr_SLR')}\t{r.get('file')}")

def get_data(path_folder: str,
             model_type: str) -> dict:
    files = os.listdir(path_folder)

    file_list = []

    for file in files:
        path = os.path.join(path_folder, file, model_type+'.yaml')
        with open(path, 'rb') as f:
            cfg = yaml.safe_load(f)
        
        layers = cfg['layers']

        layer = layers[0]
        params = layer['params']
        rate_rank = params['rate_rank']
        rate_sparsity = params['rate_sparsity']

        alpha_dict = params['alpha_dict']
        beta_dict = params['beta_dict']
        rho_dict = params['rho_dict']

        dalpha = alpha_dict['rate_decay']
        alpha_mode = alpha_dict['mode']
        dbeta = beta_dict['rate_decay']
        beta_mode = beta_dict['mode']
        rho = rho_dict['rho']

        path = os.path.join(path_folder, file, 'eval_results.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        eval_results = data['eval_test_results']
        ppl_X = eval_results['X']['ppl']
        nr_X = eval_results['nr_X'] / 1e6

        ppl_SLR = eval_results['L_with_S']['ppl']
        nr_SLR = eval_results['nr_L_with_S'] / 1e6
    # output table of different files as file | rho | dalpha | alpha_mode | dbeta | beta_mode | rate_rank | rate_sparsity
        if rate_rank == 0.15 and rate_sparsity == 0.05:
            if alpha_mode == 'adaptive' and beta_mode == 'adaptive':
                file_list.append({
                    'file': file,
                    'rho': rho,
                    'dalpha': dalpha,
                    'dbeta': dbeta,
                    'ppl_X': ppl_X,
                    'nr_X': nr_X,
                    'ppl_SLR': ppl_SLR,
                    'nr_SLR': nr_SLR
                })
    return file_list


def main(model_type: str,
         folder: str):
    
    path_folder = os.path.join(root, 'data', folder, model_type)
    file_list = get_data(path_folder, model_type)

    (A, B) = find_longest_sequences(file_list)

    best_key_dbeta, rows_dbeta, n_dbeta = A
    best_key_dalpha, rows_dalpha, n_dalpha = B

    # if best_key_dbeta is None:
    #     print("No valid groups found for dbeta ablation (check file_list contents).")
    # else:
    #     pretty_print_dbeta_ablation(best_key_dbeta, rows_dbeta)

    # print()

    # if best_key_dalpha is None:
    #     print("No valid groups found for dalpha ablation (check file_list contents).")
    # else:
    #     pretty_print_dalpha_ablation(best_key_dalpha, rows_dalpha)

    rho_sweeps = find_rho_sweeps_same_dalpha_dbeta(file_list, min_rhos=2, dedup_strategy="best_ppl_slr")
    print_rho_sweeps(rho_sweeps)


if __name__ == "__main__":
    MODEL_TYPE = 'llama_130m'
    FOLDER = 'ablation'

    main(MODEL_TYPE,
         FOLDER)
