"""This script is used to compare the convergence behavior 
of SALAAD with and without embedding layers included.
"""
import sys, os
import matplotlib.pyplot as plt
import pickle
import tikzplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layers_mapping = {
    'k': 'self_attn.k_proj',
    'q': 'self_attn.q_proj',
    'v': 'self_attn.v_proj',
    'o': 'self_attn.o_proj',
    'up': 'mlp.up_proj',
    'down': 'mlp.down_proj',
    'gate': 'mlp.gate_proj',
}

def get_matrix(path, layer_name):
    files = os.listdir(path)
    rank_files = [f for f in files if f.startswith('matrix')]
    for f in rank_files:
        LL, SS = get_lowspa_layers(os.path.join(path, f))
        for key in LL:
            if key == layer_name:
                L = LL[key].to(device)
                S = SS[key].to(device)
                return L, S
    raise ValueError(f'Layer {layer_name} not found in {path}')

def get_info(path):
    path_file = os.path.join(path, 'layer_info.pkl')
    with open(path_file, 'rb') as f:
        info = pickle.load(f)
    return info

def plot_sparsity(S_baseline: torch.Tensor,
                  S_head: torch.Tensor,
                  MODEL_TYPE: str,
                  layer_type: str,
                  path_folder) -> None:
    file_name = f'comparison_sparsity_{MODEL_TYPE}_{layer_type}'
    S_baseline = S_baseline.cpu().numpy()
    S_head = S_head.cpu().numpy()
    eps_baseline = np.max(np.abs(S_baseline)) / 100000000.0
    eps_head = np.max(np.abs(S_head)) / 100000000.0

    Sb = np.asarray(S_baseline)
    Sh = np.asarray(S_head)

    Mb = np.abs(Sb) > eps_baseline
    Mh = np.abs(Sh) > eps_head

    shared  = Mb & Mh
    added   = (~Mb) & Mh
    removed = Mb & (~Mh)

    # code: 0 background, 1 removed, 2 added, 3 shared
    code = np.zeros(Sb.shape, dtype=np.uint8)
    code[removed] = 1
    code[added] = 2
    code[shared] = 3

    plt.figure(figsize=(6, 6))
    plt.imshow(code, interpolation='nearest')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_rank(L_baseline: torch.Tensor,
            L_head: torch.Tensor,
            MODEL_TYPE: str,
            layer_type: str,
            path_folder) -> None:
    file_name = f'comp_rank_{MODEL_TYPE}_{layer_type}'
    _, s_baseline, _ = torch.linalg.svd(L_baseline, full_matrices=False)
    _, s_head, _ = torch.linalg.svd(L_head, full_matrices=False)

    s_baseline = s_baseline.cpu().numpy()
    s_head = s_head.cpu().numpy()

    k = 50 # int(0.05 * len(s_baseline))  # 默认展示前 20%

    indices = np.arange(0, k)

    plt.figure(figsize=(8, 4.8))

    # Baseline：黑色边框
    plt.bar(
        indices,
        s_baseline[:k],
        width=0.8,
        facecolor='none',
        edgecolor='black',
        linewidth=1.0,
        label='Baseline'
    )

    # Head：红色边框（完全重合）
    plt.bar(
        indices,
        s_head[:k],
        width=0.8,
        facecolor='none',
        edgecolor='red',
        linewidth=1.0,
        linestyle='-',   # 可选：再加一个区分维度
        label='Head'
    )

    # tikzplotlib.save(os.path.join(path_folder, file_name + '.tex'))
    # plt.savefig(os.path.join(path_folder, file_name + '.png'))
    plt.show()

def plot_loss(loss_baseline: list,
              loss_head: list,
              MODEL_TYPE: str,
              path_folder) -> None:
    file_name = f'comp_loss_{MODEL_TYPE}'
    # downsample for better visualization

    loss_b = np.asarray(loss_baseline)
    loss_h = np.asarray(loss_head)

    steps = len(loss_b)

    # ---- 下采样设置（主图与 inset 可分开设置）----
    stride_main = 50
    stride_inset = 50  # inset 也下采样；你可以设成 5 让 inset 更细

    # 主图下采样索引：0, stride_main, 2*stride_main, ...
    idx_main = np.arange(0, steps, stride_main)

    # ---- 绘图：主图（下采样，但 x 仍是原始 steps）----
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(idx_main, loss_b[idx_main], 'b-')
    ax.plot(idx_main, loss_h[idx_main], 'r-')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Average Loss')
    ax.legend()

    # ---- inset：放大后期收敛区间（同样下采样，且 x 语义不变）----
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
    ratio_value = 0.9
    start = int(ratio_value * steps)
    end = steps - 1

    # inset 区间内下采样索引：start, start+stride_inset, ...
    idx_inset = np.arange(start, steps, stride_inset)

    axins.plot(idx_inset, loss_b[idx_inset], 'b-')
    axins.plot(idx_inset, loss_h[idx_inset], 'r-')

    # inset 的 x 范围保持原始 Training Steps 的语义（不被压缩）
    axins.set_xlim(start, end)

    # y 轴自动放大：用后期区间的数据确定范围（可用全量后期，保证不漏尖峰）
    tail_full = np.concatenate([loss_b[start:], loss_h[start:]])
    y_min = float(tail_full.min())
    y_max = float(tail_full.max())

    # 给一点 padding，避免线贴边
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1e-12
    axins.set_ylim(y_min - pad, y_max + pad)

    # inset 里不显示刻度（需要时可改成显示）
    axins.set_xticks([])
    axins.set_yticks([])

    plt.savefig(os.path.join(path_folder, file_name + '.png'))
    tikzplotlib.save(os.path.join(path_folder, file_name + '.tex'))

    # plt.show()
    # plt.close()

def plot_embedding(rank_head: list,
                    sparsity_head: list,
                    total_rank: int,
                    total_elements: int,
                    MODEL_TYPE: str,
                    layer_type: str,
                    path_folder) -> None:
    file_name = f'comp_embed_{MODEL_TYPE}_{layer_type}'

    rank_head = [r / total_rank for r in rank_head]
    sparsity_head = [(total_elements - s) / total_elements for s in sparsity_head]

    rh = np.asarray(rank_head, dtype=float)
    sh = np.asarray(sparsity_head, dtype=float)

    n = len(rh)
    steps = np.arange(n)

    # ---------- downsampling ----------
    stride_main = 1
    idx_main = np.arange(0, n, stride_main)
    # ---------- single axis ----------
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # rank curves
    ax.plot(steps[idx_main], rh[idx_main],
            label='Rank ratio (Head)')
    # sparsity curves
    ax.plot(steps[idx_main], sh[idx_main],
            label='Sparsity (Head)')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Ratio')
    
    plt.savefig(os.path.join(path_folder, file_name + '.png'))
    tikzplotlib.save(os.path.join(path_folder, file_name + '.tex'))
    # plt.show()

def plot_layer(rank_baseline: list,
               sparsity_baseline: list,
               rank_head: list,
               sparsity_head: list,
               total_rank: int,
               total_elements: int,
               MODEL_TYPE: str,
               layer_type: str,
               path_folder) -> None:
    file_name = f'comp_layer_{MODEL_TYPE}_{layer_type}'

    rank_baseline = [r / total_rank for r in rank_baseline]
    rank_head = [r / total_rank for r in rank_head]
    sparsity_baseline = [(total_elements - s) / total_elements for s in sparsity_baseline]
    sparsity_head = [(total_elements - s) / total_elements for s in sparsity_head]

    rb = np.asarray(rank_baseline, dtype=float)
    rh = np.asarray(rank_head, dtype=float)
    sb = np.asarray(sparsity_baseline, dtype=float)
    sh = np.asarray(sparsity_head, dtype=float)

    n = len(rb)
    steps = np.arange(n)

    # ---------- downsampling ----------
    stride_main = 1
    stride_inset = 1
    zoom_ratio = 0.8

    idx_main = np.arange(0, n, stride_main)
    start = int(zoom_ratio * n)
    start = min(max(start, 0), n - 2)
    idx_inset = np.arange(start, n, stride_inset)

    # ---------- single axis ----------
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # rank curves
    ax.plot(steps[idx_main], rb[idx_main],
            label='Rank ratio (Baseline)')
    ax.plot(steps[idx_main], rh[idx_main],
            label='Rank ratio (Head)')

    # sparsity curves
    ax.plot(steps[idx_main], sb[idx_main],
            label='Sparsity (Baseline)')
    ax.plot(steps[idx_main], sh[idx_main],
            label='Sparsity (Head)')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Ratio')
    leg = ax.legend(loc='best')

    # ====================================================
    # inset 1: zoom-in for rank
    # ====================================================
    axins_rank = inset_axes(
        ax, width="38%", height="38%", loc="upper left"
    )

    axins_rank.plot(steps[idx_inset], rb[idx_inset])
    axins_rank.plot(steps[idx_inset], rh[idx_inset])

    axins_rank.set_xlim(start, n - 1)

    tail_r = np.concatenate([rb[start:], rh[start:]])
    y_min_r, y_max_r = float(tail_r.min()), float(tail_r.max())
    pad_r = 0.05 * (y_max_r - y_min_r) if y_max_r > y_min_r else 1e-12
    axins_rank.set_ylim(y_min_r - pad_r, y_max_r + pad_r)

    axins_rank.set_xticks([])
    axins_rank.set_yticks([])
    axins_rank.set_title('Rank (zoom)', fontsize=8)

    # ====================================================
    # inset 2: zoom-in for sparsity
    # ====================================================
    axins_spars = inset_axes(
        ax, width="38%", height="38%", loc="lower right"
    )

    axins_spars.plot(steps[idx_inset], sb[idx_inset])
    axins_spars.plot(steps[idx_inset], sh[idx_inset])

    axins_spars.set_xlim(start, n - 1)

    tail_s = np.concatenate([sb[start:], sh[start:]])
    y_min_s, y_max_s = float(tail_s.min()), float(tail_s.max())
    pad_s = 0.05 * (y_max_s - y_min_s) if y_max_s > y_min_s else 1e-12
    axins_spars.set_ylim(y_min_s - pad_s, y_max_s + pad_s)

    axins_spars.set_xticks([])
    axins_spars.set_yticks([])
    axins_spars.set_title('Sparsity (zoom)', fontsize=8)
    
    plt.savefig(os.path.join(path_folder, file_name + '.png'))
    tikzplotlib.save(os.path.join(path_folder, file_name + '.tex'))
    # plt.show()

def main(MODEL_TYPE: str,
         file_baseline_fp32: str,
         file_head_fp32: str,
         nr_layer: int,
         layer_type: str,
         if_plot_loss: bool=False,
         if_plot_layer: bool=False,
         if_plot_embed: bool=False,
         if_plot_dist: bool=True) -> None:
    
    path_folder = os.path.join(root, 'data', 'figures', 'comparison_embedding')
    mkdir(path_folder)

    layer_name = f'layers.{nr_layer}.{layers_mapping[layer_type]}'
    path_file_baseline_fp32 = os.path.join(root, 'data', 'baseline_fp32', MODEL_TYPE, file_baseline_fp32)
    path_file_head_fp32 = os.path.join(root, 'data', 'head_fp32', MODEL_TYPE, file_head_fp32)

    info_baseline_fp32 = get_info(path_file_baseline_fp32)
    info_head_fp32 = get_info(path_file_head_fp32)
    
    L_baseline, _ = get_matrix(path_file_baseline_fp32, layer_name)
    L_head, _ = get_matrix(path_file_head_fp32, layer_name)

    if if_plot_dist:
        plot_rank(L_baseline,
                    L_head,
                    MODEL_TYPE,
                    layer_type,
                    path_folder)

        # plot_sparsity(S_baseline,
        #             S_head,
        #             MODEL_TYPE,
        #             layer_type,
        #             path_folder)
        
    if if_plot_loss:
        plot_loss(info_baseline_fp32['avg_loss'],
                  info_head_fp32['avg_loss'],
                  MODEL_TYPE,
                  path_folder)
        
    if if_plot_layer:
        plot_layer(info_baseline_fp32[layer_name]['rank'],
                   info_baseline_fp32[layer_name]['nonzero'],
                   info_head_fp32[layer_name]['rank'],
                   info_head_fp32[layer_name]['nonzero'],
                   info_baseline_fp32[layer_name]['total_rank'][0],
                   info_baseline_fp32[layer_name]['total_elements'][0],
                   MODEL_TYPE,
                   layer_type,
                   path_folder)

    if if_plot_embed:
        plot_embedding(info_head_fp32['embed_tokens']['rank'],
                   info_head_fp32['embed_tokens']['nonzero'],
                   info_head_fp32['embed_tokens']['total_rank'][0],
                   info_head_fp32['embed_tokens']['total_elements'][0],
                   MODEL_TYPE,
                   'embed',
                   path_folder)



if __name__ == "__main__":
    MODEL_TYPE = 'llama_350m'
    nr_layer = 21
    layer_type = 'up'

    files_baseline_fp32_list = [
        '20251204_135646',
        '20251202_164626',
        '20251203_102315',
        '20251130_125959',
    ]

    files_head_fp32_list = [
        '20251204_152747',
        '20251203_144749',
        '20251204_134313'
    ]   

    for file in files_baseline_fp32_list:
        path_file = os.path.join(root, 'data', 'baseline_fp32', MODEL_TYPE, file)
        if os.path.exists(path_file):
            file_baseline_fp32 = file
        
    for file in files_head_fp32_list:
        path_file = os.path.join(root, 'data', 'head_fp32', MODEL_TYPE, file)
        if os.path.exists(path_file):
            file_head_fp32 = file
    
    main(MODEL_TYPE, 
         file_baseline_fp32,
         file_head_fp32,
         nr_layer,
         layer_type)