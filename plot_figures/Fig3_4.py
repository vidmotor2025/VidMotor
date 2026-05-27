import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from tools import calculate_statistics, calc_pvalue
from matplotlib import font_manager
from matplotlib.lines import Line2D
import argparse
from pathlib import Path

# styles
script_dir = Path(__file__).resolve().parent
font_path = script_dir / 'Arial.ttf'

matplotlib.rcParams["font.family"] = "Arial"
arial_font = font_manager.FontProperties(fname=font_path, size=10)
# models
model_ordered = ['SkateFormer', 'HD-GCN', 'FR-Head', 'MotionBERT', 'SkeleT-GCN', 'MMN', 'ProtoGCN', 'Ours']
color_models = {'SkateFormer': '#7da494', 'HD-GCN': '#eab67a', 'FR-Head': '#9E97B7', 'MotionBERT': '#6e8fb2',
                'SkeleT-GCN': '#DDBF9F', 'MMN': '#D9A7BE', 'ProtoGCN': '#9DA6B9', 'Ours': '#c16e71'}
marker_models = {'SkateFormer': '^', 'HD-GCN': 'D', 'FR-Head': 'P', 'MotionBERT': 's', 'SkeleT-GCN': 'H', 'MMN': '>',
                 'ProtoGCN': 'X', 'Ours': 'o'}
# datasets
xlabel_names_fig3 = {'SSBD-Line-Gait': 'SSBD-Line-Gait\n(binary)', 'EHE-Hands-Wave': 'EHE-Hands-Wave\n(binary)',
                     'EHE-Hands-UpDown': 'EHE-Hands-UpDown\n(binary)', 'EHE-Waist-BendL': 'EHE-Waist-BendL\n(binary)',
                     'EHE-Waist-BendR': 'EHE-Waist-BendR\n(binary)', 'EHE-Forward-Gait': 'EHE-Forward-Gait\n(binary)',
                     'EHE-Backward-Gait': 'EHE-Backward-Gait\n(binary)', 'PD-Round-Gait': 'PD-Round-Gait\n(binary)',
                     'PD-Walkway-Gait': 'PD-Walkway-Gait\n(3-class)', 'PD4T-Round-Gait': 'PD4T-Round-Gait\n(4-class)',
                     'PD4T-Leg-Agility': 'PD4T-Leg-Agility\n(4-class)'}
dataset_fig3 = list(xlabel_names_fig3.keys())
xlabel_names_fig4 = {'IRDS-Elbow-FlexionL': 'IRDS-Elbow-FlexionL\n(binary)',
                     'IRDS-Elbow-FlexionR': 'IRDS-Elbow-FlexionR\n(binary)',
                     'IRDS-Shoulder-FlexionL': 'IRDS-Shoulder-FlexionL\n(binary)',
                     'IRDS-Shoulder-FlexionR': 'IRDS-Shoulder-FlexionR\n(binary)',
                     'IRDS-Shoulder-AbductionL': 'IRDS-Shoulder-AbductionL\n(binary)',
                     'IRDS-Shoulder-AbductionR': 'IRDS-Shoulder-AbductionR\n(binary)',
                     'IRDS-Shoulder-Forward': 'IRDS-Shoulder-Forward\n(binary)',
                     'IRDS-Side-TapL': 'IRDS-Side-TapL\n(binary)',
                     'IRDS-Side-TapR': 'IRDS-Side-TapR\n(binary)',
                     'TRSP-Seated-Motion': 'TRSP-Seated-Motion\n(4-class)'}
dataset_fig4 = list(xlabel_names_fig4.keys())

metrics = {'Accuracy': 'Accuracy (%)', 'F1-score': 'F1 score (%)', 'Balanced-accuracy': 'Balanced accuracy (%)',
           'AUROC': 'AUROC'}
metric_list = list(metrics.keys())


def plot_scatter_subaxes(datasets, data_dict, model_ordered, figsize, ylabel, xlabel_dict, savepath=None):
    model_values = np.array([[data_dict[model][ds]['mean'] for ds in datasets] for model in model_ordered])
    our_values = np.array([data_dict['Ours'][ds]['mean'] for ds in datasets])
    datasets.extend(['Mean'])
    n_datasets = len(datasets)
    figsize = (figsize[0] + 0.3, figsize[1])
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=False)
    if n_datasets == 1:
        axes = [axes]
    # p-val
    our_mean = np.mean(our_values)
    compara_values = model_values[:-1, :]
    compara_means = np.mean(compara_values, axis=0)
    compara_mean_scatter = np.mean(compara_means)
    pval = calc_pvalue(our_values, compara_means)
    # scatters
    scatter_size = 40
    scatter_xcor = 0  # position on x-axis
    for i, ds in enumerate(datasets):
        ax = axes[i]
        if i == len(datasets) - 1:  # the last column
            ax.scatter(scatter_xcor, our_mean, color=color_models['Ours'], marker=marker_models['Ours'],
                       edgecolors='none', s=scatter_size, alpha=1, clip_on=False)
            ax.scatter(scatter_xcor, compara_mean_scatter, color='#8A8A8A', marker='p', s=scatter_size, alpha=1,
                       clip_on=False)
            y_values = [our_mean, compara_mean_scatter]
        else:
            y_values = []
            for j, model in enumerate(model_ordered):
                y = data_dict[model][ds]['mean']
                ax.scatter(scatter_xcor, y, facecolors=color_models[model], edgecolors='none',
                           marker=marker_models[model], s=scatter_size, alpha=0.75, clip_on=False)
                y_values.append(y)
        # y-axis
        # Each subplot calculates its own maximum tick value independently.
        if ylabel == 'AUROC':
            y_min = min(y_values) - 0.01
            y_lim_bottom = np.floor(y_min * 10) / 10
            y_lim_top = np.ceil(max(y_values) * 10) / 10
            ax.set_ylim([y_lim_bottom, y_lim_top])
            ax.set_yticks([y_lim_bottom, y_lim_top])
            ax.set_yticklabels([f"{y_lim_bottom:.1f}", f"{y_lim_top:.1f}"], fontproperties=arial_font)
        else:
            y_min = min(y_values) - 0.5
            y_lim_bottom = np.floor(y_min)
            # Slightly expand axis limits to avoid clipping edge points
            if ('F1' in ylabel and ds == 'EHE-Forward-Gait') or (
                    'Balanced' in ylabel and ds == 'IRDS-Elbow-FlexionR') or (
                    'Balanced' in ylabel and ds == 'IRDS-Shoulder-AbductionL') or (
                    'F1' in ylabel and 'IRDS-Elbow-FlexionL' in ds) or ('F1' in ylabel and 'IRDS-Side-Tap' in ds) or (
                    'F1' in ylabel and ds == 'PD-Round-Gait') or (
                    'F1' in ylabel and ds == 'Mean' and len(datasets) == 11) or (
                    'Balanced' in ylabel and ds == 'IRDS-Side-TapL'):
                y_lim_bottom -= 1
            elif ('F1' in ylabel and 'IRDS-Elbow-FlexionR' in ds) or (
                    'F1' in ylabel and ds == 'IRDS-Shoulder-AbductionL'):
                y_lim_bottom -= 2
            y_lim_top = np.ceil(max(y_values))
            ax.set_ylim([y_lim_bottom, y_lim_top])
            ax.set_yticks([y_lim_bottom, y_lim_top])
            ax.set_yticklabels([f"{int(y_lim_bottom)}", f"{int(np.ceil(y_lim_top))}"], fontproperties=arial_font)
        ax.tick_params(axis='y', pad=1)
        # x-axis
        ax.set_xticks([0])
        if ds == 'Mean':
            ax.set_xticklabels([ds], rotation=60, ha='right', multialignment='center', fontproperties=arial_font,
                               rotation_mode='anchor')
        else:
            ax.set_xticklabels([xlabel_dict[ds]], rotation=60, ha='right', multialignment='center',
                               fontproperties=arial_font, rotation_mode='anchor')
        # metric labels (show on the left side for the first axis)
        if i == 0:
            ax.set_ylabel(ylabel, fontproperties=arial_font)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # distance between subplot axes
    plt.subplots_adjust(wspace=2)
    # p-val
    if not np.isnan(pval):
        ax = plt.gca()
        line_len = 0.05
        x_right = 0.9 + line_len
        ax.plot([x_right, x_right], [our_mean, compara_mean_scatter], transform=ax.get_yaxis_transform(),
                color="black", linewidth=0.8, clip_on=False)
        if pval < 0.0001:
            ax.text(x_right + 0.1, (our_mean + compara_mean_scatter) / 2, r"$P$ < 0.0001",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontproperties=arial_font)
        else:
            ax.text(x_right + 0.1, (our_mean + compara_mean_scatter) / 2, f"$P$ = {pval:.4f}",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontproperties=arial_font, )
    datasets.remove('Mean')
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_legend(save_path):
    handles = [Line2D([0], [0], marker=marker_models[lab], color='w', markerfacecolor=color_models[lab], markersize=8,
                      label=lab) for lab in model_ordered]
    plt.figure(figsize=(8, 0.2))
    plt.legend(handles=handles, loc='center', ncol=8, frameon=False, prop=arial_font,
               handlelength=1.5, handletextpad=0.5, columnspacing=0.7)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def load_result_fig3_4(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    model_columns = [col for col in df.columns if col not in ['Dataset']]
    results = {}
    for m in metric_list:
        results[m] = {}
    dataset_list = df['Dataset'].unique().tolist()  # all datasets in result file
    result_dataset_fig3 = []  # available datasets for figure 3
    result_dataset_fig4 = []  # available datasets for figure 4
    result_info = {}
    for model in model_columns:
        model_name = model.split('_')[0]
        current_metric = [m for m in metric_list if m in model]
        assert len(current_metric) == 1
        results[current_metric[0]][model_name] = {}
        if not current_metric[0] in result_info:
            result_info[current_metric[0]] = []
        result_info[current_metric[0]].append(model_name)
        for ds in dataset_fig3:
            if ds in dataset_list:
                if not ds in result_dataset_fig3:
                    result_dataset_fig3.append(ds)
                values = df.loc[df['Dataset'] == ds, model].values
                if current_metric[0] == 'AUROC':
                    values /= 100
                results[current_metric[0]][model_name][ds] = calculate_statistics(values)
        for ds in dataset_fig4:
            if ds in dataset_list:
                if not ds in result_dataset_fig4:
                    result_dataset_fig4.append(ds)
                values = df.loc[df['Dataset'] == ds, model].values
                if current_metric[0] == 'AUROC':
                    values /= 100
                results[current_metric[0]][model_name][ds] = calculate_statistics(values)
    for k in result_info:
        result_info[k] = sorted(result_info[k], key=lambda x: model_ordered.index(x))
    return results, result_info, result_dataset_fig3, result_dataset_fig4


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot Figure 3 and Figure 4')
    parser.add_argument('--file-path', type=str, default='./results_examples/results_finetune.xlsx',
                        help='Path to the Excel file with results')
    parser.add_argument('--save-dir', type=str, default='./figures/examples/',
                        help='Directory to save output figures')
    args = parser.parse_args()
    file_path = args.file_path
    save_dir_fig3 = os.path.join(args.save_dir, 'Fig3')
    save_dir_fig4 = os.path.join(args.save_dir, 'Fig4')
    os.makedirs(save_dir_fig3, exist_ok=True)
    os.makedirs(save_dir_fig4, exist_ok=True)
    # load results
    results, result_info, result_dataset_fig3, result_dataset_fig4 = load_result_fig3_4(file_path)
    # figures
    if not len(result_dataset_fig3) == 0:
        for m in result_info:
            plot_scatter_subaxes(result_dataset_fig3, results[m], result_info[m], (5.5, 1.8), metrics[m], xlabel_names_fig3,
                                 f'{save_dir_fig3}/{m}.svg')
        plot_legend(f'{save_dir_fig3}/legend.svg')
        print(f'Fig3 -- {save_dir_fig3}')
    else:
        print('No valid datasets for Fig3.')
    if not len(result_dataset_fig4) == 0:
        for m in result_info:
            plot_scatter_subaxes(result_dataset_fig4, results[m], result_info[m], (5.5, 1.8), metrics[m], xlabel_names_fig4,
                                 f'{save_dir_fig4}/{m}.svg')
        plot_legend(f'{save_dir_fig4}/legend.svg')
        print(f'Fig4 -- {save_dir_fig4}')
    else:
        print('No valid datasets for Fig4.')

