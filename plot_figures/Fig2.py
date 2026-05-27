import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from tools import calculate_statistics, calc_pvalue
from matplotlib import font_manager
from matplotlib.patches import Patch
import argparse
from pathlib import Path

# styles
script_dir = Path(__file__).resolve().parent
font_path = script_dir / 'Arial.ttf'

matplotlib.rcParams["font.family"] = "Arial"
arial_font = font_manager.FontProperties(fname=font_path, size=7)
# models
model_ordered = ['SkateFormer', 'HD-GCN', 'FR-Head', 'MotionBERT', 'SkeleT-GCN', 'MMN', 'ProtoGCN', 'Ours']
color_models = {'SkateFormer': '#7da494', 'HD-GCN': '#eab67a', 'FR-Head': '#9E97B7', 'MotionBERT': '#6e8fb2',
                'SkeleT-GCN': '#DDBF9F', 'MMN': '#D9A7BE', 'ProtoGCN': '#9DA6B9', 'Ours': '#c16e71'}
# datasets
xlabel_names_fig2 = {'SPHERE-Stair-Gait': 'SPHERE-Stair-Gait\n(5-class)',
                     'SPHERE-Surface-Gait': 'SPHERE-Surface-Gait\n(3-class)',
                     'SPHERE-Sit': 'SPHERE-Sit\n(binary)', 'SPHERE-Stand': 'SPHERE-Stand\n(binary)',
                     'Walking-Treadmill-Gait': 'Walking-Treadmill-Gait\n(9-class)'}
dataset_fig2 = list(xlabel_names_fig2.keys())
metrics = {'Accuracy': 'Accuracy (%)', 'F1-score': 'F1 score (%)', 'Balanced-accuracy': 'Balanced accuracy (%)',
           'AUROC': 'AUROC'}
metric_list = list(metrics.keys())


def plot_bar(datasets, data_dict, model_ordered, figsize, ylabel, save_path=None):
    plot_values = np.array([[data_dict[model][ds]['mean'] for ds in datasets] for model in model_ordered])
    our_values = np.array([data_dict['Ours'][ds]['mean'] for ds in datasets])
    n_models = len(model_ordered)
    n_datasets = len(datasets)
    # bar positions
    bar_width = 0.2
    group_spacing = n_models * bar_width + bar_width
    x = np.arange(n_datasets) * group_spacing
    # figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sca_values = []
    # bars
    for i, model in enumerate(model_ordered):
        bar_positions = x + i * bar_width
        bars = plt.bar(bar_positions, plot_values[i], width=bar_width, label=model, color=color_models[model], alpha=0.9)
        for j, ds in enumerate(datasets):
            values = data_dict[model][ds]['values']
            sca_values.append(values)
            x_positions = [bar_positions[j]] * len(values)
            plt.scatter(x_positions, values, facecolors='none', edgecolors='gray', s=1, alpha=0.4, zorder=3,
                        clip_on=False)
    sca_values = np.array(sca_values)

    # average line and p-val
    ours_mean = np.mean(our_values)
    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [ours_mean, ours_mean], color=color_models['Ours'], linestyle='--', linewidth=0.8, alpha=0.8)
    compara_values = plot_values[:-1, :]
    compara_means = np.mean(compara_values, axis=0)
    compara_mean_scatter = np.mean(compara_means)
    plt.plot([xmin, xmax], [compara_mean_scatter, compara_mean_scatter], color='#8A8A8A', linestyle='--',
             linewidth=0.8, alpha=0.8)
    pval = calc_pvalue(our_values, compara_means)
    if not np.isnan(pval):
        line_len = 0.05
        x_right = 0.97 + line_len
        ax.plot([x_right, x_right], [ours_mean, compara_mean_scatter], transform=ax.get_yaxis_transform(),
                color="black", linewidth=0.8, clip_on=False)
        # p-val notation
        if pval < 0.0001:
            ax.text(x_right + 0.01, (ours_mean + compara_mean_scatter) / 2, rf"$P$ < 0.0001",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontproperties=arial_font)
        else:
            ax.text(x_right + 0.01, (ours_mean + compara_mean_scatter) / 2, rf"$P$ = {pval:.4f}",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontproperties=arial_font)
    # y-axis
    plt.ylabel(ylabel, fontproperties=arial_font)
    if ylabel == 'AUROC':
        plt.ylim(top=1.0)
        plt.yticks([0, 0.5, 1.0], fontproperties=arial_font)
    elif 'F1' in ylabel:
        plt.ylim(top=100)
        plt.yticks([0, 50, 100],fontproperties=arial_font)
    else:
        plt.ylim(20, 100)
        plt.yticks([20, 60, 100], fontproperties=arial_font)

    if n_datasets == 5 and n_models == 8:
        plt.xlim(xmin + 0.25, xmax - 0.25)
    plt.xticks(x + (n_models - 1) * bar_width / 2, [xlabel_names_fig2[d] for d in datasets], ha='center', rotation=30,
               fontproperties=arial_font)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_legend(save_path):
    handles = [Patch(facecolor=color, label=label) for label, color in color_models.items()]
    plt.figure(figsize=(5.5, 0.2))
    plt.legend(handles=handles, loc='center', ncol=len(color_models), frameon=False, prop=arial_font,
               handlelength=1.5, handletextpad=0.5, columnspacing=0.7)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def load_result_fig2(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    model_columns = [col for col in df.columns if col not in ['Dataset']]
    results = {}
    for m in metric_list:
        results[m] = {}
    dataset_list = df['Dataset'].unique().tolist()  # all datasets in result file
    result_dataset = []  # available datasets for figure
    result_info = {}
    for model in model_columns:
        model_name = model.split('_')[0]
        current_metric = [m for m in metric_list if m in model]
        assert len(current_metric) == 1
        current_metric = current_metric[0]
        results[current_metric][model_name] = {}
        if not current_metric in result_info:
            result_info[current_metric] = []
        result_info[current_metric].append(model_name)
        for ds in dataset_fig2:
            if ds in dataset_list:
                values = df.loc[df['Dataset'] == ds, model].values
                if current_metric == 'AUROC':
                    values /= 100
                if not ds in result_dataset:
                    result_dataset.append(ds)
                results[current_metric][model_name][ds] = calculate_statistics(values)
    for k in result_info:
        result_info[k] = sorted(result_info[k], key=lambda x: model_ordered.index(x))
    return results, result_info, result_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Figure 2')
    parser.add_argument('--file-path', type=str, default='./results_examples/results_finetune.xlsx',
                        help='Path to the Excel file with results')
    parser.add_argument('--save-dir', type=str, default='./figures/examples/Fig2',
                        help='Directory to save output figures')
    args = parser.parse_args()
    file_path = args.file_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # load results
    results, result_info, result_dataset = load_result_fig2(file_path)
    # figures
    if not len(result_dataset)==0:
        for m in result_info:
            plot_bar(result_dataset, results[m], result_info[m], (4.2, 2.3), metrics[m], f'{save_dir}/{m}.svg')
        plot_legend(f'{save_dir}/legend.svg')
        print(f'Fig2 -- {save_dir}')
    else:
        print('No valid datasets for Fig2.')

