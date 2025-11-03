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
        eval_dict: dict,         # 新增：评测结果字典
        target_keys: list,       # 新增：目标 key 列表（会先映射再取值）
        key_word_map: dict       # 新增：target_key -> 实际 eval key 的映射
    ) -> pd.DataFrame:
    """
    在原始每行记录的基础上，额外写入（对每个 mapped_key 复制一份）：
      - eval_key:   映射后的 key
      - ppl:        eval_dict[mapped_key]['ppl']（若不存在则 NaN）
      - n_params:   eval_dict['nr+' + mapped_key] 或 eval_dict['nr_' + mapped_key]（若不存在则 NaN）
    """
    import numpy as np
    import pandas as pd

    # -------- 预处理：把 target_keys 映射为 eval 中实际的 key，并取出 ppl / n_params --------
    mapped_infos = []
    if not isinstance(target_keys, (list, tuple)):
        target_keys = [target_keys]

    for tk in target_keys:
        mk = key_word_map.get(tk, tk)
        ppl = np.nan
        n_params = np.nan
        if isinstance(eval_dict, dict):
            # ppl: 支持 eval_dict[mk]['ppl']
            v = eval_dict.get(mk, {})
            if isinstance(v, dict):
                ppl = v.get('ppl', np.nan)
            # n_params: 支持 'nr+mk' 或 'nr_mk'
            if f"nr+{mk}" in eval_dict:
                n_params = eval_dict.get(f"nr+{mk}", np.nan)
            elif f"nr_{mk}" in eval_dict:
                n_params = eval_dict.get(f"nr_{mk}", np.nan)
        mapped_infos.append(dict(
            eval_key=str(mk),
            ppl=ppl if ppl is not None else np.nan,
            n_params=n_params if n_params is not None else np.nan,
        ))

    # 如果一个都没取到，也放一个占位，避免上游崩
    if not mapped_infos:
        mapped_infos = [dict(eval_key="unknown", ppl=np.nan, n_params=np.nan)]

    # -------- 常规逐层统计（对每个 mapped_key 复制一份） --------
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
                    # 评测信息（随 eval_key 复制）
                    eval_key=info["eval_key"],
                    ppl=float(info["ppl"]) if info["ppl"] is not None else np.nan,
                    n_params=float(info["n_params"]) if info["n_params"] is not None else np.nan,
                ))
    base = pd.DataFrame(rows)

    # -------- 扩展出 scope 视图（layer / subcomp / block / all）---------
    scope_records = []
    for r in base.itertuples(index=False):
        rdict = r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)

        # layer 作用域
        rec = dict(rdict)
        rec["scope_type"] = "layer"
        rec["scope_name"] = str(rec["layer_idx"])
        scope_records.append(rec)

        # subcomp 作用域
        rec = dict(rdict)
        rec["scope_type"] = "subcomp"
        rec["scope_name"] = str(rec["subcomp"])
        scope_records.append(rec)

        # block 作用域
        rec = dict(rdict)
        rec["scope_type"] = "block"
        rec["scope_name"] = str(rec["block"])
        scope_records.append(rec)

        # all 作用域
        rec = dict(rdict)
        rec["scope_type"] = "all"
        rec["scope_name"] = "all"
        scope_records.append(rec)

    df = pd.DataFrame(scope_records)

    # -------- 类型整理 ----------
    for c in ["rho","alpha","beta","value","ppl","n_params"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["exp_id","layer","block","subcomp","metric","scope_type","scope_name","eval_key"]:
        df[c] = df[c].astype("category")
    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce").astype("Int64")
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
    path: str = None,
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
            if path is None:
                plt.show()
            else:
                path_file = os.path.join(path, f"{scope_type}_{metric}.png")
                g.fig.set_size_inches(16, 8) 
                plt.savefig(path_file, dpi=300)
                plt.close()
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
                height=height,
                aspect=1.6,
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
    eval_order: list = None,        # 指定 x 轴 target key 的顺序（eval_key）；None 自动
    rho_whitelist=None,             # 仅保留这些 rho；None 表示全部
    height: float = 3.6,            # 每个子图高度
    max_cols: int = 6,              # Facet col_wrap
    y_range: tuple = (25, 100),          # 固定 ppl 范围，如 (0, 50)；None 自适应
    marker_size: float = 150,        # 散点大小
    marker_alpha: float = 0.9,      # 散点透明度
    cmap_name: str = "viridis",     # 连续色图，用于 n_params（深色=大）
    title_prefix: str = "",         # 图标题前缀
    label_fontsize: int = 8,        # 🔹参数文本字号
    label_offset_frac: float = 10.0, # 🔹参数文本上移（相对 y 轴跨度的比例）
    yscale: str = "linear",        # y 轴尺度，"linear" 或 "log"
    path: str = None,              # 保存路径；None 则显示不保存
):
    # ---------- 基础检查 ----------
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

    # ---------- 先把 eval_key 变为普通字符串，避免 groupby category 扩展组合 ----------
    d["eval_key"] = d["eval_key"].astype(str)

    # ---------- 聚合：对每个 (alpha,beta,rho,eval_key) 求 ppl/n_params 的均值 ----------
    # observed=True 只对“观测到”的类别组合分组，避免未来 pandas 默认变更导致警告/错误
    d = (
        d.groupby(["alpha", "beta", "rho", "eval_key"], observed=True, as_index=False)
         .agg(ppl=("ppl", "mean"), n_params=("n_params", "mean"))
    )

    # ---------- 类别准备（会创建 alpha_cat、beta_cat、rho_cat） ----------
    d, alpha_labels, beta_labels = _ensure_categories(d)

    # ---------- x 轴 eval_key 顺序 ----------
    if eval_order is None:
        # 默认按出现顺序（保序去重）
        eval_order = list(dict.fromkeys(d["eval_key"].tolist()))
    d["eval_key"] = pd.Categorical(d["eval_key"], categories=eval_order, ordered=True)

    # ---------- ρ 顺序 ----------
    rho_levels = list(np.sort(d["rho"].dropna().unique()))
    d["rho_cat"] = pd.Categorical(d["rho"], categories=rho_levels, ordered=True)

    # ---------- 参数量单位转为百万 M ----------
    d["n_params"] = d["n_params"].astype(float) / 1e6

    # ---------- 连续颜色映射：n_params 深色=大 ----------
    n_vals = d["n_params"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(n_vals) == 0:
        n_min, n_max = 0.0, 1.0
    else:
        n_min, n_max = float(n_vals.min()), float(n_vals.max())
        if n_min == n_max:
            n_max = n_min + 1.0
    norm = colors.Normalize(vmin=n_min, vmax=n_max)
    cmap = cm.get_cmap(cmap_name)

    # ---------- 形状映射：rho ----------
    marker_pool = ['o','s','^','D','P','X','v','<','>','h','*']
    if len(rho_levels) > len(marker_pool):
        reps = int(np.ceil(len(rho_levels) / len(marker_pool)))
        marker_pool = (marker_pool * reps)[:len(rho_levels)]
    rho_marker = {lvl: marker_pool[i] for i, lvl in enumerate(rho_levels)}

    # ---------- Facet 网格 ----------
    g = sns.FacetGrid(
        d,
        row="beta_cat",
        col="alpha_cat",
        margin_titles=True,
        sharey=(y_range is not None),  # 固定 y_range 时各子图共享 y
        col_wrap=max_cols if max_cols and d["alpha_cat"].nunique() > max_cols else None,
        height=height,
        aspect=1.6,
    )

    # ---------- 绘制函数（单个子图） ----------
    def _map_scatter(data, color=None, **kwargs):
        ax = plt.gca()

        # x = eval_key（分类，定位到 0..K-1）
        if hasattr(data["eval_key"], "cat"):
            cats = list(data["eval_key"].cat.categories)
        else:
            cats = sorted(data["eval_key"].unique(), key=str)
        cat_to_x = {c: i for i, c in enumerate(cats)}

        # 为不同 rho 做轻微水平偏移，避免重叠
        present_rhos = [r for r in rho_levels if r in set(data["rho_cat"].dropna().tolist())]
        K = max(1, len(present_rhos))
        max_offset = 0.18
        offsets = np.linspace(-max_offset, max_offset, K)
        rho_to_offset = {r: offsets[i] for i, r in enumerate(present_rhos)}

        # 逐点绘制
        for _, row in data.iterrows():
            x0 = cat_to_x[row["eval_key"]]
            off = rho_to_offset.get(row["rho_cat"], 0.0)
            x = x0 + off
            y = row["ppl"]
            # 颜色取决于 n_params
            cval = cmap(norm(float(row["n_params"])) if pd.notnull(row["n_params"]) else 0.0)
            # 形状取决于 rho
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
            # 在点的中间/略上方标注参数个数（单位 M，保留两位小数）
            try:
                y_min, y_max = ax.get_ylim()
                # y_offset = (y_max - y_min) * float(label_offset_frac)
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
                path_effects=[pe.withStroke(linewidth=0.5, )]  # 白色描边防遮挡
            )


        # x 轴刻度与标签（固定在分类中心）
        ax.xaxis.set_major_locator(FixedLocator(list(range(len(cats)))))
        ax.xaxis.set_major_formatter(FixedFormatter([str(c) for c in cats]))
        ax.grid(True, axis="y", alpha=0.2, zorder=0)

    g.map_dataframe(_map_scatter)

    # ---------- 统一坐标轴标题隐藏，行列标题显示 α/β ----------
    g.set_axis_labels("", "")
    try:
        g.set_titles(row_template=r"$\beta$={row_name:.6g}", col_template=r"$\alpha$={col_name:.6g}")
    except Exception:
        g.set_titles(row_template=r"$\beta$={row_name}", col_template=r"$\alpha$={col_name}")

    # ---------- Tick 可见性：左列显示 y，底行显示 x；可选固定 y 范围 ----------
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

    # ---------- 仅在最右上角子图添加 ρ 图例（marker 形状） ----------
    legend_ax = None
    if n_rows > 0 and n_cols > 0:
        legend_ax = axes[0, -1]  # 第一行最右列
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
            # title="ρ",
            loc="upper right",
            frameon=True,
            fontsize=9,
        )
        if lgd.get_title():
            lgd.get_title().set_fontsize(10)

    # ---------- 右侧添加 colorbar（参数数量，单位 M，保留两位小数） ----------
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 先压缩主图区域，为右侧空出空间
    plt.subplots_adjust(top=0.90, bottom=0.20, right=0.90)  # <-- 原来是 right=0.92

    # 在最右侧绘制 colorbar
    cbar = g.fig.colorbar(
        sm,
        ax=g.axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.02,   # 增宽一点
        pad=0.04         # 往右移一点
    )
    cbar.set_label("Nr. Prms(M)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # ---------- 标题与边距 ----------
    prefix = (title_prefix + " | ") if title_prefix else ""
    g.fig.suptitle(f"PPL and Nr. Parameters", y=0.98)

    if path is None:
        plt.show()
    else:
        path_file = os.path.join(path, f"ppl_grid.png")
        g.fig.set_size_inches(16, 8) 
        plt.savefig(path_file, dpi=300)
        plt.close()