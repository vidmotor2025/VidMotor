import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from tools import calc_pvalue
from matplotlib import font_manager
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from pathlib import Path
import argparse
from tools import calculate_statistics
import os

# styles
script_dir = Path(__file__).resolve().parent
font_path = script_dir / 'Arial.ttf'

matplotlib.rcParams["font.family"] = "Arial"
arial_font = font_manager.FontProperties(fname=font_path, size=8)
# models
model_ordered = ['SkateFormer', 'HD-GCN', 'FR-Head', 'MotionBERT', 'SkeleT-GCN', 'MMN', 'ProtoGCN', 'Ours']
color_models = {'SkateFormer': '#7da494', 'HD-GCN': '#eab67a', 'FR-Head': '#9E97B7', 'MotionBERT': '#6e8fb2',
                'SkeleT-GCN': '#DDBF9F', 'MMN': '#D9A7BE', 'ProtoGCN': '#9DA6B9', 'Ours': '#c16e71'}
# dataset
tasks = ['Arising from Chair', 'Leg Agility', 'Gait', 'Freezing of Gait', ]
metrics = {'Accuracy': 'Accuracy (%)', 'Balanced-accuracy': 'Balanced accuracy (%)', 'F1-score': 'F1 score (%)',
           'AUROC': 'AUROC'}
metric_list = list(metrics.keys())
center_list = ['center-1', 'center-2', 'center-3', 'center-4']
y_label_dict = {'Arising from Chair': 'PDMC-Arising-from-Chair\n(5-class)',
                'Leg Agility': 'PDMC-Leg-Agility\n(5-class)', 'Gait': 'PDMC-Gait\n(5-class)',
                'Freezing of Gait': 'PDMC-Freezing-of-Gait\n(5-class)'}


def subplots_box_vertical(data_dict, figsize=(10, 6), savepath=None):
    total_groups = len(data_dict)  # 4 metrics (4 sub-figures)
    group_size = 8  # 8 models
    fig, axes = plt.subplots(1, total_groups, figsize=figsize, sharey=False)
    model_ordered.reverse()
    tasks_copy = tasks.copy()
    tasks_copy.reverse()
    # sub-figures
    for ax_id, metric in enumerate(metric_list):
        i = 0
        y_pos_interval = 0.5
        # box styles
        box_line_width = 0.5
        box_line_color = 'black'
        box_line_alpha = 0.8
        # axes
        ax = axes[ax_id]
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        mean_values = {}
        task_centers = []
        task_labels = []
        for act in tasks_copy:
            start_i = i
            for model in model_ordered:
                if model not in mean_values:
                    mean_values[model] = []
                # data
                x = np.array(data_dict[metric][act][model])  # data of 4 centers
                mean_values[model].append(np.mean(x))
                pos = i + y_pos_interval  # box position
                bplot = ax.boxplot(x, positions=[pos], patch_artist=True, widths=0.7, showfliers=False, vert=False)
                # interval line
                if model == 'Ours':
                    ax.axhline(y=pos - y_pos_interval, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)
                # bos styles
                for patch in bplot['boxes']:
                    patch.set_facecolor(color_models[model])
                    patch.set_alpha(0.8)
                    patch.set_linewidth(box_line_width)
                for whisker in bplot['whiskers']:
                    whisker.set_linewidth(box_line_width)
                    whisker.set_color(box_line_color)
                    whisker.set_alpha(box_line_alpha)
                for cap in bplot['caps']:
                    cap.set_linewidth(box_line_width)
                    cap.set_color(box_line_color)
                    cap.set_alpha(box_line_alpha)
                for median in bplot['medians']:
                    median.set_linewidth(box_line_width)
                    median.set_color(box_line_color)
                    median.set_alpha(box_line_alpha)
                # scatter
                y_jitter = np.random.normal(pos, 0.04, size=len(x))
                face_rgba = mcolors.to_rgba(color_models[model], alpha=0.9)
                edge_rgba = mcolors.to_rgba(color_models[model], alpha=1)
                ax.scatter(x, y_jitter, facecolor=face_rgba, edgecolor=edge_rgba, s=8, linewidths=0.5, zorder=10)
                i += 1
            end_i = i - 1
            center = (start_i + end_i) / 2 + y_pos_interval
            task_centers.append(center)
            task_labels.append(y_label_dict[act])
        # p-val
        compara_values = []
        for model, model_means in mean_values.items():
            if model != 'Ours':
                compara_values.append(model_means)
        compara_values = np.array(compara_values)
        compara_means = np.mean(compara_values, axis=0)
        our_means = mean_values['Ours']
        pval = calc_pvalue(our_means, compara_means)
        # mean scatter
        our_mean_val = np.mean(our_means)
        compara_mean_val = np.mean(compara_means)
        ymin, ymax = plt.ylim()
        mean_y = ymin - y_pos_interval
        ax.scatter(our_mean_val, mean_y, color=color_models['Ours'], marker='o', s=25, alpha=1)
        ax.scatter(compara_mean_val, mean_y, color='#8A8A8A', marker='p', s=25, alpha=1)
        # p-val notation
        y_line = mean_y - y_pos_interval
        ax.plot([compara_mean_val, our_mean_val], [y_line, y_line], color='black', linewidth=0.8)
        if pval < 0.0001:
            ax.text((compara_mean_val + our_mean_val) / 2, y_line - y_pos_interval * 0.3, rf"$P$ < 0.0001", ha='center',
                    va='top', fontproperties=arial_font)
        else:
            ax.text((compara_mean_val + our_mean_val) / 2, y_line - y_pos_interval * 0.3, rf"$P$ = {pval:.4f}",
                    ha='center', va='top', fontproperties=arial_font)
        # y-axis
        if ax_id == 0:
            ax.set_yticks(task_centers + [mean_y])
            ax.set_yticklabels(task_labels + ['Mean'], fontproperties=arial_font, multialignment='center')
        else:
            ax.set_yticks([])
            ax.tick_params(axis='y', length=0)
        # x-axis
        if metric == 'AUROC':
            ax.set_xticks([0.45, 0.7, 0.95])
        elif metric == 'Accuracy':
            ax.set_xticks([25, 50, 75])
        elif metric == 'Balanced-accuracy':
            ax.set_xticks([20, 50, 80])
        elif metric == 'F1-score':
            ax.set_xticks([25, 50, 75])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(metrics[metric], fontproperties=arial_font)
        for label in ax.get_xticklabels():
            label.set_fontproperties(arial_font)
    plt.subplots_adjust(wspace=0.2)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_std_bars(plot_data, figsize=(3, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    group_idx = np.arange(len(tasks))
    n_bars = len(model_ordered)
    bar_h = 0.1
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_h * 1.2
    tasks_copy = tasks.copy()
    tasks_copy.reverse()
    for i, t in enumerate(tasks_copy):
        for m, model in enumerate(model_ordered):
            ax.barh(group_idx[i] + offsets[m], plot_data[model][t], height=bar_h, color=color_models[model],
                    alpha=0.9)
    # calculate mean values
    mean_vals = []
    for ds in tasks:
        mean_vals.append(np.array([plot_data[model][ds] for model in model_ordered if model != 'Ours']).mean())
    mean_vals = np.array(mean_vals)
    mean_val = mean_vals.mean()
    mean_ours = np.array([plot_data['Ours'][ds] for ds in tasks]).mean()
    # position for mean scatters
    mean_y = group_idx[0] + offsets[0] * 1.35
    # plot mean values
    ax.scatter(mean_ours, mean_y, color=color_models['Ours'], marker='o', s=25, alpha=1)
    ax.scatter(mean_val, mean_y, color='#8A8A8A', marker='p', s=25, alpha=1)
    # axis
    plt.ylim(-0.8, 3.5)
    ax.xaxis.set_ticks_position('top')
    ticks = [i / 100 for i in range(0, 13, 4)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontproperties=arial_font)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel('Inter-center SD of AUROC', fontproperties=arial_font)
    ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_legend(save_path):
    handles = [Patch(facecolor=color, label=label) for label, color in color_models.items()]
    plt.figure(figsize=(6.5, 0.2))
    plt.legend(handles=handles, loc='center', ncol=len(color_models), frameon=False, prop=arial_font,
               handlelength=1.5, handletextpad=0.5, columnspacing=0.7)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def load_result_fig5(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    results = {}
    for task in tasks:
        results[task] = {}
        for model in model_ordered:
            results[task][model] = {}
            for metric in metric_list:
                row = df[(df["Task"] == task) & (df.iloc[:, 0] == model)]
                results[task][model][metric] = {}
                for center in center_list:
                    if metric == 'AUROC':
                        results[task][model][metric][center] = row.iloc[0][f'{metric}_{center}'] / 100
                    else:
                        results[task][model][metric][center] = row.iloc[0][f'{metric}_{center}']
    plot_std = {}
    for model in model_ordered:
        if not model in plot_std:
            plot_std[model] = {}
        for ds in tasks:
            data_model = np.array(list(results[ds][model]['AUROC'].values()))
            sta_val = calculate_statistics(data_model)['std']
            plot_std[model][ds] = sta_val
    return results, plot_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Figure 5')
    parser.add_argument('--file-path', type=str, default='./results_examples/results_multicenter.xlsx',
                        help='Path to the Excel file with results')
    parser.add_argument('--save-dir', type=str, default='./figures/examples/Fig5',
                        help='Directory to save output figures')
    args = parser.parse_args()
    file_path = args.file_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # load results
    results, plot_std = load_result_fig5(file_path)
    # figures
    box_plot_data = {}
    for metric in metric_list:
        box_plot_data[metric] = {}
        for task in tasks:
            box_plot_data[metric][task] = {}
            for model in model_ordered:
                box_plot_data[metric][task][model] = []
                for center in center_list:
                    box_plot_data[metric][task][model].append(results[task][model][metric][center])
    # box plot
    subplots_box_vertical(box_plot_data, (9, 6), f'{save_dir}/a_boxplot.svg')
    # bars
    plot_std_bars(plot_std, (1.7, 5.2), f'{save_dir}/b_std.svg')
    plot_legend(f'{save_dir}/legend.svg')
    print(f'Fig5 -- {save_dir}')
