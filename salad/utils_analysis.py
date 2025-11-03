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
        data: dict,  # {layer_name: {'rank': [...], 'sparsity': [...], 'loss': [...]}, ...}
        layer_names: list
    ) -> pd.DataFrame:
    rows = []
    for name in layer_names:
        if name not in data:
            continue
        layer_idx, block, sub = parse_from_name(name)
        for metric_key in ('rank', 'loss', 'sparsity'):
            val = get_layer_stats(data, name, metric_key)
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
            ))
    base = pd.DataFrame(rows)

    # -------- 新增：扩展出 scope 视图（layer / subcomp / block / all）---------
    # 复制每行数据，分别标注 scope_type 与 scope_name
    scope_records = []
    for r in base.itertuples(index=False):
        rdict = r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)  # 兼容性
        # layer 作用域：按层号聚合
        rec = dict(rdict)
        rec["scope_type"] = "layer"
        rec["scope_name"] = str(rec["layer_idx"])
        scope_records.append(rec)
        # subcomp 作用域：按子组件聚合
        rec = dict(rdict)
        rec["scope_type"] = "subcomp"
        rec["scope_name"] = str(rec["subcomp"])
        scope_records.append(rec)
        # block 作用域：按块名聚合
        rec = dict(rdict)
        rec["scope_type"] = "block"
        rec["scope_name"] = str(rec["block"])
        scope_records.append(rec)
        # all 作用域：全局（所有层一起）
        rec = dict(rdict)
        rec["scope_type"] = "all"
        rec["scope_name"] = "all"
        scope_records.append(rec)

    df = pd.DataFrame(scope_records)

    # -------- 保持/完善数据类型 ----------
    for c in ["rho","alpha","beta","value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["exp_id","layer","block","subcomp","metric","scope_type","scope_name"]:
        df[c] = df[c].astype("category")
    # layer_idx 保持可空整数，便于排序与缺失兼容
    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce").astype("Int64")

    # 给 scope_type 固定顺序，方便 Facet 稳定
    df["scope_type"] = df["scope_type"].cat.set_categories(["all","block","subcomp","layer"], ordered=True)

    return df


def _ensure_categories(df: pd.DataFrame):
    """把 alpha/beta/rho 变成排序好的类别，确保 Facet 网格顺序稳定可控。"""
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
    # rho 用作 x 轴分类（小提琴分组）
    df["rho_cat"] = pd.Categorical(df["rho"], ordered=True)

    # （可选）确保 scope_name 也是 category，避免字符串/数值混排导致的顺序抖动
    if "scope_name" in df.columns and not pd.api.types.is_categorical_dtype(df["scope_name"]):
        df["scope_name"] = df["scope_name"].astype("category")

    return df, alpha_labels, beta_labels

def plot_violin_grid(
    df: pd.DataFrame,
    *,
    scope_type: str,              # 'all' | 'layer' | 'subcomp' | 'block'
    scope_name=None,              # 指定具体对象时=旧逻辑；None 时=把该 scope_type 的所有对象合并到一张图
    metrics=("loss","rank","sparsity"),
    rho_whitelist=None,
    max_cols=6,
    save_prefix="violin_grids",
    height=3.6,
    y_ranges: dict = None,        # e.g. dict(loss=(0,20), rank=(0,1), sparsity=(0,1))
):

    y_ranges = y_ranges or dict(loss=(0,10), rank=(0,1), sparsity=(0,0.6))

    # 1) 过滤 scope_type
    d = df.copy()
    d = d[d["scope_type"] == scope_type]

    # 2) 限定 rho
    if rho_whitelist is not None:
        rho_whitelist = set(float(x) for x in rho_whitelist)
        d = d[d["rho"].apply(lambda x: float(x) in rho_whitelist)]
    if d.empty:
        return

    # 3) 类别准备（alpha/beta/rho）
    d, alpha_labels, beta_labels = _ensure_categories(d)

    # ================= 合并模式（只针对当前 scope_type 自己） =================
    if scope_name is None and scope_type in ("layer", "subcomp", "block"):
        # 横轴类别：仅来自“当前”的 scope_type
        if scope_type == "layer":
            d["unit"] = d["layer_idx"].astype("Int64").astype(str).map(lambda s: f"L{s}")
            def _layer_key(u):
                try: return int(u[1:])
                except: return 10**9
            units = sorted(d["unit"].astype(str).unique(), key=_layer_key)

        elif scope_type == "subcomp":
            # 目标顺序与显示标签
            desired_order = ["down", "gate", "up", "o", "v", "k", "q"]

            # 规范化子组件名到短标签（仅用于显示与分组，不改原数据）
            def _norm_subcomp(s: str) -> str:
                s = str(s).lower().replace("-", "_")
                if s.endswith("_proj"):
                    s = s[:-5]
                # 常见同义词映射
                if s in ("out", "output", "outp"): s = "o"
                if s in ("value", "val"):          s = "v"
                if s in ("key",):                   s = "k"
                if s in ("query",):                 s = "q"
                # 其余保留（如 down/gate/up/o/v/k/q）
                return s

            d["unit"] = d["subcomp"].astype(str).map(_norm_subcomp)

            # 只保留数据里实际出现的类别，但按 desired_order 的顺序排列
            present = [u for u in desired_order if u in set(d["unit"].astype(str).unique())]
            extras = [u for u in d["unit"].astype(str).unique() if u not in present]
            units = present + sorted(extras, key=str)

        else:  # block
            d["unit"] = d["block"].astype(str)
            units = sorted(d["unit"].astype(str).unique(), key=str)

        d["unit"] = pd.Categorical(d["unit"], categories=units, ordered=True)

        # hue：rho 全局一致顺序 + 固定颜色映射
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
                col_wrap=max_cols if max_cols and dd["alpha_cat"].nunique() > max_cols else None,
                height=height,
                legend=False,                 # 不生成子图图例
                palette=rho_palette           # 固定颜色映射
            )

            # === 去掉 axis label（x/y label），保留刻度文字 ===
            g.set_axis_labels("", "")
            try:
                g.set_titles(row_template=r"$\beta$={row_name:.6g}", col_template=r"$\alpha$={col_name:.6g}")
            except Exception:
                g.set_titles(row_template=r"$\beta$={row_name}", col_template=r"$\alpha$={col_name}")

            # === 仅左列显示 y ticklabel，底行显示 x ticklabel ===
            y_limits = y_ranges.get(metric)
            axes = g.axes if g.axes is not None else np.array([])
            axes = np.atleast_2d(axes)
            n_rows, n_cols = axes.shape

            for r in range(n_rows):
                for c in range(n_cols):
                    ax = axes[r, c] if n_cols > 0 else None
                    if ax is None:
                        continue

                    # 网格线
                    ax.grid(True, axis="y", alpha=0.2)

                    # 只在最左列显示 y 的刻度文字
                    ax.tick_params(axis="y", which="both", labelleft=(c == 0))

                    # 只在最底行显示 x 的刻度文字
                    is_bottom = (r == n_rows - 1)
                    ax.tick_params(axis="x", which="both", labelbottom=is_bottom)

                    if is_bottom:
                        # 取当前分类轴的刻度位置与文字
                        ticks = ax.get_xticks()
                        labels = [t.get_text() for t in ax.get_xticklabels()]
                        # 用 FixedLocator/FixedFormatter 固定（避免警告）
                        ax.xaxis.set_major_locator(FixedLocator(ticks))
                        ax.xaxis.set_major_formatter(FixedFormatter(labels))
                        # 旋转 + 右对齐
                        for lab in ax.get_xticklabels():
                            lab.set_rotation(60)
                            lab.set_ha("right")

                    # y 轴范围
                    if y_limits is not None:
                        ax.set_ylim(*y_limits)

            # === 仅在最右上角子图添加颜色图例（ρ → 颜色） ===
            axes2d = np.atleast_2d(g.axes) if g.axes is not None else np.array([])
            legend_ax = None
            if axes2d.size > 0:
                legend_ax = axes2d[0, -1]  # 第一行最右列
            if legend_ax is None:
                # 兜底：找第一个非空轴
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
            plt.show()

        return

    # ================= 旧逻辑：指定了 scope_name 或 scope_type == 'all' =================
    # （保持你的原始行为：每个对象一张图，x=rho）
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
                row="beta_cat", col="alpha_cat",
                margin_titles=True, sharey=False,
                col_wrap=max_cols if max_cols and dd["alpha_cat"].nunique() > max_cols else None,
                height=3.0
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
            plt.show()