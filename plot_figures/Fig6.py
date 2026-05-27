import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import pandas as pd
from matplotlib import font_manager
from tools import calculate_statistics
from matplotlib.patches import Patch
import os
from Fig2 import load_result_fig2
from Fig3_4 import load_result_fig3_4
from Fig5 import load_result_fig5
from Fig5 import tasks as dataset_multicenter

# styles
script_dir = Path(__file__).resolve().parent
font_path = script_dir / 'Arial.ttf'

matplotlib.rcParams["font.family"] = "Arial"
arial_font = font_manager.FontProperties(fname=font_path, size=8)
# models
color_mean = '#8A8A8A'
color_mean_multicenter='#8B4513'
model_ordered = ['SkateFormer', 'HD-GCN', 'FR-Head', 'MotionBERT', 'SkeleT-GCN', 'MMN', 'ProtoGCN', 'Ours']
color_models = {'SkateFormer': '#7da494', 'HD-GCN': '#eab67a', 'FR-Head': '#9E97B7', 'MotionBERT': '#6e8fb2',
                'SkeleT-GCN': '#DDBF9F', 'MMN': '#D9A7BE', 'ProtoGCN': '#9DA6B9', 'Ours': '#c16e71'}
color_centers = {'Arising from Chair': '#CC9981', 'Leg Agility': '#DEBFAA', 'Gait': '#B7B8A3',
                 'Freezing of Gait': '#A5A58C'}
color_centers_legend = {'PDMC-Arising-from-Chair': '#CC9981', 'PDMC-Leg-Agility': '#DEBFAA', 'PDMC-Gait': '#B7B8A3',
                        'PDMC-Freezing-of-Gait': '#A5A58C'}
labels_multicenter = list(color_centers_legend.keys())


def load_result_ours(file_path):
    # 10% training data
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    result_ours = dict(zip(df["Dataset"], df["Ours_AUROC"]))
    return result_ours


def load_result_runtime(file_path):
    df_runtime = pd.read_excel(file_path, sheet_name='Sheet1')
    model_columns = [col for col in df_runtime.columns if col not in ['Dataset']]
    dataset_list = df_runtime['Dataset'].tolist()  # all datasets in runtime file
    result = {}
    for model in model_columns:
        model_name = model.split('_')[0]
        for ds in dataset_list:
            if not ds in result:
                result[ds] = {}
            result[ds][model_name] = df_runtime.loc[df_runtime['Dataset'] == ds, model].values[0]
    return result, dataset_list


def cal_result_four_scenarios(plot_data, result_all, result_ours, dataset_all, multicenter):
    ours_full_data_results = {'Abnormality recognition': {}, 'Clinical diagnosis and severity grading': {},
                              'Physical rehabilitation assessment': {}, }
    for model in model_ordered:
        if model == 'Ours':
            # results for ours (full percentage of training data)
            ours_full_data_results['Abnormality recognition'][model] = (
                np.array([result_all['abnormal']['AUROC'][model][ds]['mean'] for ds in dataset_all['abnormal']])).mean()
            ours_full_data_results['Clinical diagnosis and severity grading'][model] = (np.array(
                [result_all['diag_rehab']['AUROC'][model][ds]['mean'] for ds in dataset_all['diagnosis']])).mean()
            ours_full_data_results['Physical rehabilitation assessment'][model] = (np.array(
                [result_all['diag_rehab']['AUROC'][model][ds]['mean'] for ds in dataset_all['rehabilitation']])).mean()
            if multicenter:
                ours_full_data_results.update({'Multi-center application': {
                    model: (np.array([np.mean(np.array(list(result_all[ds][model]['AUROC'].values()))) for ds in
                                      dataset_multicenter])).mean()}})
        else:
            plot_data['Abnormality recognition'][model] = (
                np.array([result_all['abnormal']['AUROC'][model][ds]['mean'] for ds in dataset_all['abnormal']])).mean()
            plot_data['Clinical diagnosis and severity grading'][model] = (np.array(
                [result_all['diag_rehab']['AUROC'][model][ds]['mean'] for ds in dataset_all['diagnosis']])).mean()
            plot_data['Physical rehabilitation assessment'][model] = (np.array(
                [result_all['diag_rehab']['AUROC'][model][ds]['mean'] for ds in dataset_all['rehabilitation']])).mean()
            if multicenter:
                plot_data['Multi-center application'][model] = (
                    np.array([np.mean(np.array(list(result_all[ds][model]['AUROC'].values()))) for ds in
                              dataset_multicenter])).mean()
    # results for our model (using 10% of training data)
    plot_data['Abnormality recognition']['Ours'] = (
            np.array([result_ours[ds] for ds in dataset_all['abnormal']]) / 100).mean()
    plot_data['Clinical diagnosis and severity grading']['Ours'] = (
            np.array([result_ours[ds] for ds in dataset_all['diagnosis']]) / 100).mean()
    plot_data['Physical rehabilitation assessment']['Ours'] = (
            np.array([result_ours[ds] for ds in dataset_all['rehabilitation']]) / 100).mean()
    if multicenter:
        plot_data['Multi-center application']['Ours'] = (
                np.array([result_ours[ds] for ds in labels_multicenter]) / 100).mean()
    return plot_data, ours_full_data_results


def cal_runtime_four_scenarios(plot_data, time_data, dataset_all, dataset_list, multicenter):
    for model in model_ordered:
        plot_data['Abnormality recognition'][model] = np.array(
            [time_data[ds][model] for ds in dataset_all['abnormal'] if ds in dataset_list]).mean()
        plot_data['Clinical diagnosis and severity grading'][model] = np.array(
            [time_data[ds][model] for ds in dataset_all['diagnosis'] if ds in dataset_list]).mean()
        plot_data['Physical rehabilitation assessment'][model] = np.array(
            [time_data[ds][model] for ds in dataset_all['rehabilitation'] if ds in dataset_list]).mean()
        if multicenter:
            plot_data['Multi-center application'][model] = np.array(
                [time_data[ds][model] for ds in labels_multicenter if ds in dataset_list]).mean()
    return plot_data


def plot_bars(plot_data, group_names, figsize=(6, 6), color_dict=color_models, bar_width=0.15, xlim=None, ylim=None,
              ylabel=None, save_path=None, add_lines=None,color_mean_line='#8A8A8A'):
    n_groups = len(plot_data)
    group_idx = np.arange(n_groups)
    n_bars = len(color_dict)
    bar_w = bar_width
    half = bar_w * n_bars / 2
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w * 1.2
    margin = 0.05
    # figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for g, group in enumerate(plot_data):
        # bars
        group_val = []
        for j, bar_name in enumerate(plot_data[group]):
            ax.bar(group_idx[g] + offsets[j], plot_data[group][bar_name], width=bar_w * 0.9, color=color_dict[bar_name],
                   alpha=0.9)
            if not bar_name == 'Ours':
                group_val.append(plot_data[group][bar_name])
        # draw dashed lines for each group (mean values)
        group_val = np.array(group_val)
        left = group_idx[g] - half - margin
        right = group_idx[g] + half + margin
        mean_val = group_val.mean()
        ax.hlines(mean_val, left, right, color=color_mean_line, linestyle='--', linewidth=1, alpha=0.8)
        if add_lines:
            ax.hlines(add_lines[group]['Ours'], left, right, color=color_models['Ours'], linestyle='--', linewidth=1,
                      alpha=0.8)

    # axis
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
        if ylim[1] > 10:
            if ylim[1] % 40 == 0:
                ticks = [t for t in range(ylim[0], ylim[1] + 40, 40)]
            elif ylim[1] % 50 == 0:
                ticks = [t for t in range(ylim[0], ylim[1] + 50, 50)]
            else:
                ticks = [t for t in range(ylim[0], ylim[1], 10)]
        elif ylim[1] > 0.5:
            ticks = [t / 10 for t in range(int(ylim[0] * 10), int(ylim[1] * 10) + 1)]
        else:
            ticks = [t / 100 for t in range(int(ylim[0] * 100), int(ylim[1] * 100) + 5, 5)]
        plt.yticks(ticks, fontproperties=arial_font)
    plt.xticks(group_idx, group_names, ha='center', fontproperties=arial_font)
    if ylabel:
        plt.ylabel(ylabel, fontproperties=arial_font)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_radar(dataset, plot_data, figsize=(3, 3), save_path=None):
    num_datasets = len(dataset)
    max_scales = {}
    for ds in dataset:
        values = [scores[ds] for scores in plot_data.values()]
        max_scales[ds] = np.ceil(max(values) * 100 / 5) * 0.05
    # Standardize the data to the scale interval by scale
    normalized_data = {}
    for key, scores in plot_data.items():
        normalized_data[key] = {}
        for ds in dataset:
            min_val = 0
            max_val = max_scales[ds]
            normalized_data[key][ds] = (scores[ds] - min_val) / (max_val - min_val)
    angles = np.linspace(np.pi / 2, 5 / 2 * np.pi, num_datasets, endpoint=False)
    # figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)
    ax.grid(False)
    ax.set_rticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    # Draw radial lines and outer circle labels
    for angle, dataset_name in zip(angles, dataset):
        ax.plot([angle, angle], [0, 1], color='grey', alpha=0.8, linewidth=0.7)
        ax.text(angle, 1.15, f'{(max_scales[dataset_name]):.1f}', ha='center', va='center', fontproperties=arial_font)
    colors = {'Ours': '#c16e71', 'Mean': color_mean}
    # Plot the data
    for index, data_dict in plot_data.items():
        color = colors[index]
        values = [normalized_data[index][k] for k in dataset]
        angles_extended = np.concatenate([angles, [angles[0] + 2 * np.pi]])
        values_extended = np.concatenate([values, [values[0]]])
        ax.plot(angles_extended, values_extended, linewidth=1.5, color=color, alpha=0.9)
        ax.fill(angles_extended, values_extended, color=color, alpha=0.05, )
        ax.scatter(angles, values, color=color, s=2, alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_bar_legend(color_dict, figsize, save_path, ncol=None):
    if not ncol:
        ncol = len(color_dict)
    handles = [Patch(facecolor=color, label=label)
               for label, color in color_dict.items()]
    # reorder
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    order = []
    for c in range(ncol):
        for r in range(nrow):
            idx = r * ncol + c
            if idx < n:
                order.append(idx)
    handles = [handles[i] for i in order]
    plt.figure(figsize=figsize)
    plt.legend(handles=handles, loc='center', ncol=ncol, frameon=False, prop=arial_font,
               handlelength=1.5, handletextpad=0.5, columnspacing=0.7)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot figures')
    parser.add_argument('--file-path-all', type=str, default='./results_examples/results_finetune.xlsx',
                        help='Excel file containing results trained on the full dataset.')
    parser.add_argument('--file-path-ours', type=str, default='./results_examples/results_finetune_10%training.xlsx',
                        help='Excel file containing results trained on 10% of the dataset.')
    parser.add_argument('--file-path-multicenter', default='./results_examples/results_multicenter.xlsx',
                        help='Multi-center results trained on the full dataset (set to None if unavailable).')
    parser.add_argument('--file-path-multicenter-ours',
                        default='./results_examples/results_multicenter_10%training.xlsx',
                        help='Multi-center results trained on 10% of the dataset (set to None if unavailable).')
    parser.add_argument('--file-path-runtime', type=str,
                        default='./results_examples/results_finetune_runtime.xlsx',
                        help='Excel file containing runtime results.')
    parser.add_argument('--save-dir', type=str, default='./figures/examples/Fig6',
                        help='Directory for saving output figures.')
    args = parser.parse_args()

    file_path_all = args.file_path_all
    file_path_ours = args.file_path_ours
    file_path_multicenter = args.file_path_multicenter
    file_path_multicenter_ours = args.file_path_multicenter_ours
    file_path_runtime = args.file_path_runtime
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    '''data preparation'''
    plot_four_scenarios = {'Abnormality recognition': {}, 'Clinical diagnosis and severity grading': {},
                           'Physical rehabilitation assessment': {}, }
    plot_runtime_four_scenarios = {'Abnormality recognition': {}, 'Clinical diagnosis and severity grading': {},
                                   'Physical rehabilitation assessment': {}}
    # load results
    result_fig2, _, dataset_abnormal = load_result_fig2(file_path_all)
    result_fig34, _, dataset_diagnosis, dataset_rehabilitation = load_result_fig3_4(file_path_all)
    dataset_all = {'abnormal': dataset_abnormal, 'diagnosis': dataset_diagnosis,
                   'rehabilitation': dataset_rehabilitation}
    result_all = {'abnormal': result_fig2, 'diag_rehab': result_fig34, }
    result_ours = load_result_ours(file_path_ours)
    result_runtime, runtime_dataset_list = load_result_runtime(file_path_runtime)

    # figure labels
    scenarios_labels = ['Abnormality recognition', 'Clinical diagnosis and\nseverity grading',
                        'Physical rehabilitation\nassessment', ]
    # multi-center results
    multicenter_result_exist = False
    if file_path_multicenter != 'None':
        multicenter_result_exist = True
        result_fig5, plot_multicenter_std = load_result_fig5(file_path_multicenter)
        result_all.update(result_fig5)
        # update data dicts
        plot_four_scenarios['Multi-center application'] = {}
        plot_runtime_four_scenarios['Multi-center application'] = {}
        plot_multicenter = {'Mean': {}, 'Ours': {}}
        # replace the results (std) with 10% training
        df_multicenter = pd.read_excel(file_path_multicenter_ours, sheet_name='Sheet1')
        for ds in dataset_multicenter:
            data_ours_multicenter = df_multicenter.loc[df_multicenter['Task'] == ds, :].values.tolist()[0]
            data_ours_multicenter = np.array(data_ours_multicenter[-4:]) / 100
            sta_val = calculate_statistics(data_ours_multicenter)['std']
            plot_multicenter_std['Ours'][ds] = sta_val
        # update figure labels
        scenarios_labels.append('Multi-center application')

    '''figures'''
    # fig-a: comparison approaches across four scenarios
    plot_four_scenarios, ours_full_result = cal_result_four_scenarios(plot_four_scenarios, result_all, result_ours,
                                                                      dataset_all, multicenter_result_exist)
    plot_bars(plot_four_scenarios, scenarios_labels, (6, 1.6), color_models, 0.09, (-0.45, 3.45), (0.6, 1.0), 'AUROC',
              f'{save_dir}/a_four_scenarios.svg', ours_full_result)

    # fig-b: runtime of comparison approaches across four scenarios
    plot_runtime_four_scenarios = cal_runtime_four_scenarios(plot_runtime_four_scenarios, result_runtime, dataset_all,
                                                             runtime_dataset_list, multicenter_result_exist)
    plot_bars(plot_runtime_four_scenarios, scenarios_labels, (6, 1.6), color_models, 0.09, (-0.45, 3.45), (0, 160),
              'Inference time (ms)', f'{save_dir}/b_four_scenarios_runtime.svg')

    # fig-c: 'rehabilitation' scenario
    plot_rehabilitation = {'Mean': {}, 'Ours': {}}
    for ds in dataset_rehabilitation:
        plot_rehabilitation['Ours'][ds] = result_ours[ds] / 100
        data_rehab = np.array(
            [result_fig34['AUROC'][model][ds]['mean'] for model in model_ordered if not model == 'Ours'])
        plot_rehabilitation['Mean'][ds] = data_rehab.mean()
    plot_radar(dataset_rehabilitation, plot_rehabilitation, (1.8, 1.8),
               f'{save_dir}/c_rehabilitation_radar.svg')

    # fig-d: an example of 'rehabilitation' scenario
    if 'IRDS-Elbow-FlexionR' in dataset_rehabilitation:
        tmp_dict = {}
        for model in model_ordered:
            if model == 'Ours':  # all training data
                tmp_our_full = {
                    'IRDS-Elbow-FlexionR': {'Ours': result_fig34['AUROC'][model]['IRDS-Elbow-FlexionR']['mean']}}
            else:
                tmp_dict[model] = result_fig34['AUROC'][model]['IRDS-Elbow-FlexionR']['mean']
        tmp_dict['Ours'] = result_ours['IRDS-Elbow-FlexionR'] / 100  # 10% training data
        plot_rehabilitation_IRDS = {'IRDS-Elbow-FlexionR': tmp_dict}
        plot_bars(plot_rehabilitation_IRDS, ['IRDS-Elbow-FlexionR'], (1.4, 1.6), color_models, 0.09, (-0.45, 0.45),
                  (0.4, 1.0), 'AUROC', f'{save_dir}/d_rehabilitation_bar.svg', tmp_our_full)
    # runtime
    if 'IRDS-Elbow-FlexionR' in runtime_dataset_list:
        plot_runtime_IRDS = {'IRDS-Elbow-FlexionR': {}}
        for model in model_ordered:
            plot_runtime_IRDS['IRDS-Elbow-FlexionR'][model] = result_runtime['IRDS-Elbow-FlexionR'][model]
        plot_bars(plot_runtime_IRDS, ['IRDS-Elbow-FlexionR'], (1.4, 1.6), color_models, 0.09, (-0.45, 0.45), (0, 150),
                  'Inference time (ms)', f'{save_dir}/d_rehabilitation_runtime.svg')

    # fig-e: 'multi-center' scenario
    if multicenter_result_exist:
        plot_multicenter = {'Mean': {}, 'Ours': {}}
        for ds, ds_label in zip(dataset_multicenter, labels_multicenter):
            plot_multicenter['Ours'][ds] = result_ours[ds_label] / 100
            data_multicenter = [np.array(list(result_fig5[ds][model]['AUROC'].values())).mean() for model in
                                model_ordered if not model == 'Ours']
            data_multicenter = np.array(data_multicenter)
            plot_multicenter['Mean'][ds] = data_multicenter.mean()
        plot_radar(dataset_multicenter, plot_multicenter, (1.8, 1.8), f'{save_dir}/e_multicenter_radar.svg')

    # fig-f: an example of 'multi-center' scenario
    if multicenter_result_exist:
        plot_multicenter_AFC = {'Arising from Chair': {}}
        tmp_dict = {}
        for model in model_ordered:
            if model == 'Ours':
                tmp_our_full = {
                    'Arising from Chair': {'Ours': np.array(list(result_fig5['Arising from Chair'][model]['AUROC'].values())).mean()}}
            else:
                tmp_dict[model] = np.array(list(result_fig5['Arising from Chair'][model]['AUROC'].values())).mean()
        tmp_dict['Ours'] = result_ours['PDMC-Arising-from-Chair'] / 100
        plot_multicenter_AFC['Arising from Chair'] = tmp_dict
        plot_bars(plot_multicenter_AFC, ['PDMC-Arising-from-Chair'], (1.45, 1.6), color_models, 0.09, (-0.45, 0.45),
                  (0.6, 0.9), 'AUROC', f'{save_dir}/f_multicenter_bar.svg', tmp_our_full)
        # runtime
        if 'PDMC-Arising-from-Chair' in runtime_dataset_list:
            plot_runtime_AFC = {'PDMC-Arising-from-Chair': {}}
            for model in model_ordered:
                plot_runtime_AFC['PDMC-Arising-from-Chair'][model] = result_runtime['PDMC-Arising-from-Chair'][model]
            plot_bars(plot_runtime_AFC, ['PDMC-Arising-from-Chair'], (1.45, 1.6), color_models, 0.09, (-0.45, 0.45),
                      (0, 160), 'Inference time (ms)', f'{save_dir}/f_multicenter_runtime.svg')

    # fig-g: SD results of multi-center
    if multicenter_result_exist:
        plot_bars(plot_multicenter_std, model_ordered, (6, 1.5), color_centers, 0.15, (-0.45, 7.45), (0, 0.15),
                  'Inter-center SD\nof AUROC', f'{save_dir}/g_multicenter_std_bar.svg',None,color_mean_multicenter)

    # legends
    plot_bar_legend(color_models, (3.7, 0.35), f'{save_dir}/legend_comparison.svg', 4)
    plot_bar_legend(color_centers_legend, (5.2, 0.2), f'{save_dir}/legend_multicenter.svg')
    print(f'Fig6 -- {save_dir}')
