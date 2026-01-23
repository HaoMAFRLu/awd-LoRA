"""The collection of analysis utilities for salad.
"""
import os, sys
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patches import Patch
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

def get_loss_row(file: str, 
                 data_type: str, 
                 eval_results: dict, 
                 header: list,
                 key_word_map: dict) -> list:
    """
    Get a row of loss statistics for the model.
    Args:
        file: Name of the file containing the statistics.
        eval_results: Evaluation results dictionary.
    Returns:
        A list with loss statistics.
    """
    row = [file, data_type, 'loss']
    for key in header:
        if key in key_word_map and key_word_map[key] in eval_results and eval_results[key_word_map[key]] is not None:
            _key = key_word_map[key]
            value = eval_results[_key]['avg_loss'][-1]
            if isinstance(value, float):
                if 'nr_'+_key in eval_results:
                    nr = eval_results['nr_'+_key]
                    row.append(f"{value:.4f}({nr/1000000:.2f}M)")
                else:
                    row.append(f"{value:.4f}")
            elif isinstance(value, str):   # Handle case where value is 'N/A'
                row.append(value)
        else:
            row.append('N/A')
    return row

def get_ppl_row(file: str, 
                data_type: str, 
                eval_results: dict, 
                header: list,
                key_word_map: dict) -> list:
    """
    Get a row of perplexity statistics for the model.
    Args:
        file: Name of the file containing the statistics.
        data_type: Type of data (e.g., 'train', 'test').
        eval_results: Evaluation results dictionary.
    Returns:
        A list with perplexity statistics.
    """
    row = [file, data_type, 'ppl']
    for key in header:
        if key in key_word_map and key_word_map[key] in eval_results and eval_results[key_word_map[key]] is not None:
            value = eval_results[key_word_map[key]]['ppl']
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, str):   # Handle case where value is 'N/A'
                row.append(value)
        else:
            row.append('N/A')
    return row

def get_acc_row(file: str, 
                data_type: str, 
                eval_results: dict, 
                header: list,
                key_word_map: dict) -> list:
    """
    Get a row of accuracy statistics for the model.
    Args:
        file: Name of the file containing the statistics.
        data_type: Type of data (e.g., 'train', 'test').
        eval_results: Evaluation results dictionary.
    Returns:
        A list with accuracy statistics.
    """
    row = [file, data_type, 'accuracy']
    for key in header:
        if key in key_word_map:
            row.append(f"{eval_results[key_word_map[key]]['correct']}/{eval_results[key_word_map[key]]['total']}({100.0*eval_results[key_word_map[key]]['accuracy']:.1f}%)")
        else:
            row.append('N/A')
    return row

def parse_from_name(name: str):
    m = re.search(r"layers\.(\d+)", name)
    layer_idx = int(m.group(1)) if m else None

    # block: self_attn / mlp
    if "self_attn" in name:
        block = "attn"
    elif "mlp" in name:
        block = "mlp"
    else:
        block = "other"

    # subcomp: q/k/v/o/up/gate/down 
    if "q_proj" in name:
        sub = "q_proj"
    elif "k_proj" in name:
        sub = "k_proj"
    elif "v_proj" in name:
        sub = "v_proj"
    elif "o_proj" in name:
        sub = "o_proj"
    elif "up_proj" in name:
        sub = "up_proj"
    elif "gate_proj" in name:
        sub = "gate_proj"
    elif "down_proj" in name:
        sub = "down_proj"
    else:
        sub = "other"

    return layer_idx, block, sub

def get_layer_stats(data: dict,
                    layer_name: str,
                    metric: str) -> float:
    """
    Get statistics for a specific layer and metric.
    Args:
        data: Dictionary containing layer data.
        layer_name: Name of the layer to extract.
        metric: Metric to extract (e.g., 'loss', 'rank', 'sparsity').
    Returns:
        The metric value for the specified layer.
    """
    if metric == 'loss':
        return data[layer_name][metric][-1]
    elif metric == 'rank':
        return data[layer_name][metric][-1]/data[layer_name]['total_rank'][-1]
    elif metric == 'sparsity':
        return data[layer_name]['nonzero'][-1]/data[layer_name]['total_elements'][-1]

def build_item(
        *,
        exp_id, 
        rho, 
        alpha, 
        beta,
        data: dict,              # {layer_name: {'rank': [...], 'sparsity': [...], 'loss': [...]}, ...}
        layer_names: list,
        eval_dict: dict,         # 
        target_keys: list,       #
        key_word_map: dict       # 
    ) -> pd.DataFrame:
    """
    在原始每行记录的基础上，额外写入（对每个 mapped_key 复制一份）：
      - eval_key:   映射后的 key
      - ppl:        eval_dict[mapped_key]['ppl']（若不存在则 NaN）
      - n_params:   eval_dict['nr+' + mapped_key] 或 eval_dict['nr_' + mapped_key]（若不存在则 NaN）
    """
    import numpy as np
    import pandas as pd

    mapped_infos = []
    if not isinstance(target_keys, (list, tuple)):
        target_keys = [target_keys]

    for tk in target_keys:
        mk = key_word_map.get(tk, tk)
        ppl = np.nan
        n_params = np.nan
        if isinstance(eval_dict, dict):
            v = eval_dict.get(mk, {})
            if isinstance(v, dict):
                ppl = v.get('ppl', np.nan)
            if f"nr+{mk}" in eval_dict:
                n_params = eval_dict.get(f"nr+{mk}", np.nan)
            elif f"nr_{mk}" in eval_dict:
                n_params = eval_dict.get(f"nr_{mk}", np.nan)
        mapped_infos.append(dict(
            eval_key=str(mk),
            ppl=ppl if ppl is not None else np.nan,
            n_params=n_params if n_params is not None else np.nan,
        ))

    if not mapped_infos:
        mapped_infos = [dict(eval_key="unknown", ppl=np.nan, n_params=np.nan)]

    rows = []
    for name in layer_names:
        if name not in data:
            continue
        layer_idx, block, sub = parse_from_name(name)
        for metric_key in ('rank', 'loss', 'sparsity'):
            val = get_layer_stats(data, name, metric_key)
            for info in mapped_infos:
                rows.append(dict(
                    exp_id=exp_id,
                    rho=float(rho), 
                    alpha=float(alpha), 
                    beta=float(beta),
                    layer=name,
                    layer_idx=layer_idx,
                    block=block,
                    subcomp=sub,
                    metric=metric_key,
                    value=float(val),
                    eval_key=info["eval_key"],
                    ppl=float(info["ppl"]) if info["ppl"] is not None else np.nan,
                    n_params=float(info["n_params"]) if info["n_params"] is not None else np.nan,
                ))
    base = pd.DataFrame(rows)

    scope_records = []
    for r in base.itertuples(index=False):
        rdict = r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)

        rec = dict(rdict)
        rec["scope_type"] = "layer"
        rec["scope_name"] = str(rec["layer_idx"])
        scope_records.append(rec)

        rec = dict(rdict)
        rec["scope_type"] = "subcomp"
        rec["scope_name"] = str(rec["subcomp"])
        scope_records.append(rec)

        rec = dict(rdict)
        rec["scope_type"] = "block"
        rec["scope_name"] = str(rec["block"])
        scope_records.append(rec)

        rec = dict(rdict)
        rec["scope_type"] = "all"
        rec["scope_name"] = "all"
        scope_records.append(rec)

    df = pd.DataFrame(scope_records)

    for c in ["rho","alpha","beta","value","ppl","n_params"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["exp_id","layer","block","subcomp","metric","scope_type","scope_name","eval_key"]:
        df[c] = df[c].astype("category")
    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce").astype("Int64")
    df["scope_type"] = df["scope_type"].cat.set_categories(["all","block","subcomp","layer"], ordered=True)

    return df


def _ensure_categories(df: pd.DataFrame):
    df = df.copy()
    for c in ["alpha", "beta", "rho"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def to_cat(col):
        uniq = np.sort(df[col].dropna().unique())
        labels = [f"{v:.6g}" for v in uniq]
        cat = pd.Categorical(df[col].map({u: u for u in uniq}), categories=uniq, ordered=True)
        return cat, labels

    df["alpha_cat"], alpha_labels = to_cat("alpha")
    df["beta_cat"],  beta_labels  = to_cat("beta")
    df["rho_cat"] = pd.Categorical(df["rho"], ordered=True)

    if "scope_name" in df.columns and not pd.api.types.is_categorical_dtype(df["scope_name"]):
        df["scope_name"] = df["scope_name"].astype("category")

    return df, alpha_labels, beta_labels

def plot_violin_grid(
    df: pd.DataFrame,
    *,
    scope_type: str,              # 
    scope_name=None,              #
    metrics=("loss","rank","sparsity"),
    rho_whitelist=None,
    max_cols=6,
    save_prefix="violin_grids",
    height=3.6,
    y_ranges: dict = None,        # e.g. dict(loss=(0,20), rank=(0,1), sparsity=(0,1))
    path: str = None,
):

    y_ranges = y_ranges or dict(loss=(0,10), rank=(0,1), sparsity=(0,0.6))

    d = df.copy()
    d = d[d["scope_type"] == scope_type]

    if rho_whitelist is not None:
        rho_whitelist = set(float(x) for x in rho_whitelist)
        d = d[d["rho"].apply(lambda x: float(x) in rho_whitelist)]
    if d.empty:
        return

    d, alpha_labels, beta_labels = _ensure_categories(d)

    if scope_name is None and scope_type in ("layer", "subcomp", "block"):
        if scope_type == "layer":
            d["unit"] = d["layer_idx"].astype("Int64").astype(str).map(lambda s: f"L{s}")
            def _layer_key(u):
                try: return int(u[1:])
                except: return 10**9
            units = sorted(d["unit"].astype(str).unique(), key=_layer_key)

        elif scope_type == "subcomp":
            desired_order = ["down", "gate", "up", "o", "v", "k", "q"]

            def _norm_subcomp(s: str) -> str:
                s = str(s).lower().replace("-", "_")
                if s.endswith("_proj"):
                    s = s[:-5]
                if s in ("out", "output", "outp"): s = "o"
                if s in ("value", "val"):          s = "v"
                if s in ("key",):                   s = "k"
                if s in ("query",):                 s = "q"
                return s

            d["unit"] = d["subcomp"].astype(str).map(_norm_subcomp)

            present = [u for u in desired_order if u in set(d["unit"].astype(str).unique())]
            extras = [u for u in d["unit"].astype(str).unique() if u not in present]
            units = present + sorted(extras, key=str)

        else:  # block
            d["unit"] = d["block"].astype(str)
            units = sorted(d["unit"].astype(str).unique(), key=str)

        d["unit"] = pd.Categorical(d["unit"], categories=units, ordered=True)

        rho_levels = np.sort(d["rho"].dropna().unique())
        d["rho_cat"] = pd.Categorical(d["rho"], categories=rho_levels, ordered=True)
        base_colors = sns.color_palette("tab10", n_colors=len(rho_levels))
        rho_palette = {lvl: base_colors[i] for i, lvl in enumerate(rho_levels)}

        for metric in metrics:
            dd = d[d["metric"] == metric]
            if dd.empty:
                continue
            
            g = sns.catplot(
                data=dd,
                x="unit", y="value",
                row="beta_cat", col="alpha_cat",
                hue="rho_cat",
                kind="violin",
                dodge=True, cut=0, inner="quartile",
                margin_titles=True, sharey=False,
                height=height,
                legend=False,
                palette=rho_palette,
            )


            g.set_axis_labels("", "")
            try:
                g.set_titles(row_template=r"$\beta$={row_name:.6g}", col_template=r"$\alpha$={col_name:.6g}")
            except Exception:
                g.set_titles(row_template=r"$\beta$={row_name}", col_template=r"$\alpha$={col_name}")

            y_limits = y_ranges.get(metric)
            axes = g.axes if g.axes is not None else np.array([])
            axes = np.atleast_2d(axes)
            n_rows, n_cols = axes.shape

            for r in range(n_rows):
                for c in range(n_cols):
                    ax = axes[r, c] if n_cols > 0 else None
                    if ax is None:
                        continue

                    ax.grid(True, axis="y", alpha=0.2)

                    ax.tick_params(axis="y", which="both", labelleft=(c == 0))

                    is_bottom = (r == n_rows - 1)
                    ax.tick_params(axis="x", which="both", labelbottom=is_bottom)

                    if is_bottom:
                        ticks = ax.get_xticks()
                        labels = [t.get_text() for t in ax.get_xticklabels()]
                        ax.xaxis.set_major_locator(FixedLocator(ticks))
                        ax.xaxis.set_major_formatter(FixedFormatter(labels))
                        for lab in ax.get_xticklabels():
                            lab.set_rotation(60)
                            lab.set_ha("right")

                    if y_limits is not None:
                        ax.set_ylim(*y_limits)

            axes2d = np.atleast_2d(g.axes) if g.axes is not None else np.array([])
            legend_ax = None
            if axes2d.size > 0:
                legend_ax = axes2d[0, -1]  #
            if legend_ax is None:
                for _ax in axes2d.flat:
                    if _ax is not None:
                        legend_ax = _ax
                        break

            if legend_ax is not None:
                handles = [Patch(facecolor=rho_palette[lvl], edgecolor="black", label=str(lvl))
                           for lvl in rho_levels]
                lgd = legend_ax.legend(
                    handles, [str(lvl) for lvl in rho_levels],
                    # title="ρ",
                    loc="upper right",
                    frameon=True,
                    fontsize=9,
                )
                if lgd.get_title():
                    lgd.get_title().set_fontsize(10)

            plt.subplots_adjust(top=0.90, bottom=0.18, right=0.92)
            g.fig.suptitle(f"{metric} | scope={scope_type}", y=0.98)
            if path is None:
                plt.show()
            else:
                path_file = os.path.join(path, f"{scope_type}_{metric}.png")
                g.fig.set_size_inches(16, 8) 
                plt.savefig(path_file, dpi=300)
                plt.close()
        return

    if scope_type in ("layer", "subcomp", "block"):
        if scope_name is not None:
            scope_iter = [(scope_name, d[d["scope_name"].astype(str) == str(scope_name)])]
        else:
            scope_iter = []
            for name in d["scope_name"].cat.categories if hasattr(d["scope_name"], "cat") else sorted(d["scope_name"].unique(), key=str):
                dn = d[d["scope_name"].astype(str) == str(name)]
                if not dn.empty:
                    scope_iter.append((name, dn))
    else:
        scope_iter = [("all", d)]

    for name, dsub in scope_iter:
        if dsub.empty:
            continue
        for metric in metrics:
            dd = dsub[dsub["metric"] == metric]
            if dd.empty:
                continue

            g = sns.FacetGrid(
                dd,
                row="beta_cat",
                col="alpha_cat",
                margin_titles=True, sharey=False,
                height=height, aspect=1.6,
            )

            def _map_violin(data, color=None, **kwargs):
                sns.violinplot(
                    data=data,
                    x="rho_cat", y="value",
                    inner="quartile", cut=0,
                )

            g.map_dataframe(_map_violin)
            g.set_axis_labels("rho", metric)
            try:
                g.set_titles(row_template=r"$\beta$={row_name:.6g}", col_template=r"$\alpha$={col_name:.6g}")
            except Exception:
                g.set_titles(row_template=r"$\beta$={row_name}", col_template=r"$\alpha$={col_name}")

            y_limits = y_ranges.get(metric)
            for ax in g.axes.flat:
                if ax is None:
                    continue
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, axis="y", alpha=0.2)
                if y_limits is not None:
                    ax.set_ylim(*y_limits)

            plt.subplots_adjust(top=0.88)
            g.fig.suptitle(f"{metric.upper()} | scope: {scope_type}({name})")
            if path is None:
                plt.show()
            else:
                path_file = os.path.join(path, f"{scope_type}_{metric}.png")
                g.fig.set_size_inches(16, 8) 
                plt.savefig(path_file, dpi=300)
                plt.close()


def plot_ppl_grid(
    df: pd.DataFrame,
    *,
    eval_order: list = None,         #
    rho_whitelist=None,              # 
    height: float = 3.6,             # 
    max_cols: int = 6,               # 
    is_plot_bar: bool = False,       
    y_range: tuple = (25, 100),      #
    marker_size: float = 50,        #
    marker_alpha: float = 0.9,       # 
    cmap_name: str = "viridis",      #  
    title_prefix: str = "",          # 
    label_fontsize: int = 6,         # 
    label_offset_frac: float = 10.0, # 
    yscale: str = "linear",          # 
    path: str = None,                # 
    short_name_map: dict = None,     # 
    max_label_len: int = 12,         # 
    dedup_suffix: str = "_{n}",      # 
):
    d = df.copy()
    needed = {"alpha", "beta", "rho", "eval_key", "ppl", "n_params"}
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必要列: {missing}")

    # ---------- 过滤 rho ----------
    if rho_whitelist is not None:
        rho_set = set(float(x) for x in rho_whitelist)
        d = d[d["rho"].apply(lambda x: float(x) in rho_set)]
    if d.empty:
        print("[WARN] 过滤后数据为空。")
        return

    d["eval_key"] = d["eval_key"].astype(str)

    def _truncate_if_needed(s: str) -> str:
        if max_label_len is None:
            return s
        return (s if len(s) <= max_label_len else (s[:max(1, max_label_len - 1)] + "…"))

    if short_name_map is None:
        short_name_map = {}
    unique_orig = pd.Index(d["eval_key"].unique()).tolist()
    base_map = {}
    for orig in unique_orig:
        if orig in short_name_map and short_name_map[orig] is not None:
            base_map[orig] = str(short_name_map[orig])
        else:
            base_map[orig] = _truncate_if_needed(orig)

    short_to_origs = {}
    for orig in unique_orig:
        s = base_map[orig]
        short_to_origs.setdefault(s, []).append(orig)

    final_map = {}
    for short, orig_list in short_to_origs.items():
        if len(orig_list) == 1:
            final_map[orig_list[0]] = short
        else:
            for idx, orig in enumerate(orig_list, start=1):
                if idx == 1:
                    final_map[orig] = short
                else:
                    final_map[orig] = short + dedup_suffix.format(n=idx)

    d["eval_key_short"] = d["eval_key"].map(final_map)

    d = (
        d.groupby(["alpha", "beta", "rho", "eval_key_short"], observed=True, as_index=False)
         .agg(ppl=("ppl", "mean"), n_params=("n_params", "mean"))
    )

    def _ensure_categories(df_in: pd.DataFrame):
        df_out = df_in.copy()
        alpha_vals = np.sort(df_out["alpha"].dropna().unique())
        beta_vals  = np.sort(df_out["beta"].dropna().unique())
        df_out["alpha_cat"] = pd.Categorical(df_out["alpha"], categories=alpha_vals, ordered=True)
        df_out["beta_cat"]  = pd.Categorical(df_out["beta"],  categories=beta_vals,  ordered=True)
        return df_out, alpha_vals, beta_vals

    d, alpha_labels, beta_labels = _ensure_categories(d)

    if eval_order is None:
        eval_order_short = list(dict.fromkeys(d["eval_key_short"].tolist()))
    else:
        seen = set()
        eval_order_short = []
        for key in eval_order:
            key = str(key)
            short = final_map.get(key, key)
            if short not in seen:
                eval_order_short.append(short)
                seen.add(short)
        for key in d["eval_key_short"].tolist():
            if key not in seen:
                eval_order_short.append(key)
                seen.add(key)

    d["eval_key_short"] = pd.Categorical(d["eval_key_short"], categories=eval_order_short, ordered=True)

    rho_levels = list(np.sort(d["rho"].dropna().unique()))
    d["rho_cat"] = pd.Categorical(d["rho"], categories=rho_levels, ordered=True)

    d["n_params"] = d["n_params"].astype(float) / 1e6

    n_vals = d["n_params"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(n_vals) == 0:
        n_min, n_max = 0.0, 1.0
    else:
        n_min, n_max = float(n_vals.min()), float(n_vals.max())
        if n_min == n_max:
            n_max = n_min + 1.0
    norm = colors.Normalize(vmin=n_min, vmax=n_max)
    cmap = cm.get_cmap(cmap_name)

    marker_pool = ['o','s','^','D','P','X','v','<','>','h','*']
    if len(rho_levels) > len(marker_pool):
        reps = int(np.ceil(len(rho_levels) / len(marker_pool)))
        marker_pool = (marker_pool * reps)[:len(rho_levels)]
    rho_marker = {lvl: marker_pool[i] for i, lvl in enumerate(rho_levels)}

    g = sns.FacetGrid(
        d,
        row="beta_cat",
        col="alpha_cat",
        margin_titles=True,
        sharey=(y_range is not None),
        height=height,
        aspect=1.6,
    )

    def _map_scatter(data, color=None, **kwargs):
        ax = plt.gca()

        if hasattr(data["eval_key_short"], "cat"):
            cats = list(data["eval_key_short"].cat.categories)
        else:
            cats = sorted(data["eval_key_short"].unique(), key=str)
        cat_to_x = {c: i for i, c in enumerate(cats)}

        present_rhos = [r for r in rho_levels if r in set(data["rho_cat"].dropna().tolist())]
        K = max(1, len(present_rhos))
        max_offset = 0.30
        offsets = np.linspace(-max_offset, max_offset, K)
        rho_to_offset = {r: offsets[i] for i, r in enumerate(present_rhos)}

        for _, row in data.iterrows():
            x0 = cat_to_x[row["eval_key_short"]]
            off = rho_to_offset.get(row["rho_cat"], 0.0)
            x = x0 + off
            y = row["ppl"]
            cval = cmap(norm(float(row["n_params"])) if pd.notnull(row["n_params"]) else 0.0)
            marker = rho_marker.get(row["rho_cat"], 'o')
            ax.scatter(
                x, y,
                s=marker_size,
                c=[cval],
                marker=marker,
                alpha=marker_alpha,
                edgecolors="black",
                linewidths=0.5,
                zorder=3
            )
            try:
                y_offset = label_offset_frac
            except Exception:
                y_offset = 0.0

            ppl_val = float(row["ppl"]) if pd.notnull(row["ppl"]) else np.nan
            param_val = float(row["n_params"]) if pd.notnull(row["n_params"]) else np.nan
            label = f"{ppl_val:.1f}\n{param_val:.0f}" if not np.isnan(ppl_val) and not np.isnan(param_val) else "NaN"

            ax.text(
                x, y + y_offset,
                label,
                ha="center", 
                va="bottom",
                fontsize=label_fontsize, 
                weight="light",
                linespacing=1.2, 
                color="black",
                zorder=4,
                path_effects=[pe.withStroke(linewidth=0.5)]
            )

        ax.xaxis.set_major_locator(FixedLocator(list(range(len(cats)))))
        ax.xaxis.set_major_formatter(FixedFormatter([str(c) for c in cats]))
        ax.grid(True, axis="y", alpha=0.2, zorder=0)

    g.map_dataframe(_map_scatter)

    g.set_axis_labels("", "")
    try:
        g.set_titles(row_template=r"$\beta$={row_name:.6g}", col_template=r"$\alpha$={col_name:.6g}")
    except Exception:
        g.set_titles(row_template=r"$\beta$={row_name}", col_template=r"$\alpha$={col_name}")

    axes = g.axes if g.axes is not None else np.array([])
    axes = np.atleast_2d(axes)
    n_rows, n_cols = axes.shape if axes.size else (0, 0)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if ax is None:
                continue
            ax.tick_params(axis="y", which="both", labelleft=(c == 0))
            is_bottom = (r == n_rows - 1)
            ax.tick_params(axis="x", which="both", labelbottom=is_bottom)
            if is_bottom:
                for lab in ax.get_xticklabels():
                    lab.set_rotation(60)
                    lab.set_ha("right")
            if y_range is not None:
                ax.set_ylim(*y_range)
            ax.set_yscale(yscale)

    legend_ax = None
    if n_rows > 0 and n_cols > 0:
        legend_ax = axes[0, -1]  
    if legend_ax is None:
        for _ax in axes.flat:
            if _ax is not None:
                legend_ax = _ax
                break
    if legend_ax is not None:
        handles = [
            Line2D([0], [0],
                   marker=rho_marker[lvl], color="black",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=7, linestyle="None", label=str(lvl))
            for lvl in rho_levels
        ]
        lgd = legend_ax.legend(
            handles, [str(lvl) for lvl in rho_levels],
            loc="upper right",
            frameon=True,
            fontsize=9,
        )
        if lgd.get_title():
            lgd.get_title().set_fontsize(10)

    plt.subplots_adjust(top=0.90, bottom=0.15, right=0.90)

    if is_plot_bar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = g.fig.colorbar(
            sm,
            ax=g.axes.ravel().tolist(),
            orientation="vertical",
            fraction=0.02,
            pad=0.04
        )
        cbar.set_label("Nr. Prms(M)", fontsize=10)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    prefix = (title_prefix + " | ") if title_prefix else ""
    g.fig.suptitle(f"PPL and Nr. Parameters", y=0.98)

    if path is None:
        plt.show()
    else:
        path_file = os.path.join(path, f"ppl_grid.png")
        g.fig.set_size_inches(32, 8)
        plt.savefig(path_file, dpi=300)
        plt.close()