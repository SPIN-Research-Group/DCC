#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from scipy import stats
import json
import os
import zipfile
from datetime import datetime
import argparse

def get_csv_filename_from_sheet_name(sheet_name):
    """Map sheet names to corresponding CSV files in results_with_GPU_and_kernel directory."""
    # Mapping: sheet_name -> CSV filename
    sheet_to_csv = {
        'Attacc_GEMV': '../results/@attacc-gemv_single_with_GPU.csv',
        'HBMPIM-GEMV': '../results/@hbmpim-gemv_single_with_GPU.csv',
        'Attacc_Attention': '../results/@attacc-attn_single_with_GPU.csv',
        'Attacc_Red': '../results/@attacc-red_single_with_GPU.csv',
        'HBM_Red': '../results/@hbmpim-red_single_with_GPU.csv',
        'HBM_VA': '../results/@hbmpim-va_single_with_GPU.csv',
        'HBM_RELU': '../results/@hbmpim-relu_single_with_GPU.csv',
    }
    return sheet_to_csv.get(sheet_name)

def build_filter_conditions(df, filters):
    conditions = []
    for col, values in filters.items():
        if col in df.columns:
            values_list = values if isinstance(values, list) else [values]
            if len(values_list) == 1:
                conditions.append(df[col] == values_list[0])
            else:
                conditions.append(df[col].isin(values_list))
    return conditions

def add_bar_text(ax, positions, heights, show_numbers, ylimit=None, bar_label_offset=0.1, bar_label_fontsize=8):
    if not show_numbers:
        return
    for pos, height in zip(positions, heights):
        if ylimit is not None and height + bar_label_offset > ylimit:
            text_y = ylimit - bar_label_offset
            color='black'
        else:
            text_y = height + bar_label_offset
            color = 'black'
        ax.text(pos, text_y, f'{height:.2f}', ha='center', va='center', 
                fontsize=bar_label_fontsize, fontweight='bold', color=color, rotation=0)

def plot_prediction_results_excel(plot_path="../figures/", sheet_name=None, max_groups=5, save=True, filter_config=None, show_numbers=True, ax=None, is_combined=False, bar_colors=None):

    if bar_colors is None:
        bar_colors = ["#D3D3D3", "#696969"]

    csv_path = get_csv_filename_from_sheet_name(sheet_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    sheet_config = filter_config.get(sheet_name, {}) if filter_config else {}
    xlabel_fontsize = sheet_config.get('xlabel_fontsize', 6)
    ylabel_fontsize = sheet_config.get('ylabel_fontsize', 10)
    show_ylabel = sheet_config.get('show_ylabel', True)
    xlabel_rotation = sheet_config.get('xlabel_rotation', 20)
    bar_label_offset = sheet_config.get('bar_label_offset', 0.1)
    bar_width = sheet_config.get('bar_width', 0.25)
    bar_label_fontsize = sheet_config.get('bar_label_fontsize', 8)
    legend_fontsize = sheet_config.get('legend_fontsize', 8)
    legend_height = sheet_config.get('legend_height', 1.15)
    show_legend = sheet_config.get('show_legend', True)
    legend_loc = sheet_config.get('legend_loc', 'upper right')
    double_row_batch_size = sheet_config.get('double_row_batch_size', False)
    subplots_adjust_bottom = sheet_config.get('subplots_adjust_bottom', 0.25)
    batch_label_y_position = sheet_config.get('batch_label_y_position', -0.08)
    draw_batch_borders = sheet_config.get('draw_batch_borders', False)


    config_fig_width = sheet_config.get('fig_width', None)
    config_fig_height = sheet_config.get('fig_height', 3)

    has_gpu = 'GPU_time(ms)' in df.columns
    required_base = ['our_time(ms)', 'baseline(ms)']
    
    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in sheet '{sheet_name}'")

    if filter_config and sheet_name in filter_config:
        config = filter_config[sheet_name]
        filters = config.get('filters', {})
        groupby_cols = config.get('groupby', [])


        is_gemv_type = '#head' in df.columns and 'batch_size' in filters and '#head' in filters
        
        if is_gemv_type:
            batch_sizes = filters['batch_size'] if isinstance(filters['batch_size'], list) else [filters['batch_size']]
            heads = filters['#head'] if isinstance(filters['#head'], list) else [filters['#head']]

            modified_filters = {**filters, 'batch_size': batch_sizes, '#head': heads}
            conditions = build_filter_conditions(df, modified_filters)
            filtered = df[pd.concat(conditions, axis=1).all(axis=1)].copy() if conditions else df.copy()
            filtered['original_batch_size'] = filtered['batch_size']
            filtered['original_head'] = filtered['#head']

        else:
            conditions = build_filter_conditions(df, filters)
            filtered = df[pd.concat(conditions, axis=1).all(axis=1)].copy() if conditions else df.copy()

        
        if groupby_cols:
            plot_df = filtered.groupby(groupby_cols).first().reset_index()
        else:
            plot_df = filtered
    else:
        row_indices = list(range(min(max_groups, len(df))))
        plot_df = df.iloc[row_indices].copy()

    grp_cols = [c for c in ['batch_size', '#head', 'input_dim', 'output_dim', 'vec_len'] if c in df.columns]

    include_batch_size = False
    include_input_dim = False
    ylimit = 2.5
    label_format = 'standard'
    if filter_config and sheet_name in filter_config:
        config = filter_config[sheet_name]
        include_batch_size = config.get('include_batch_size_in_label', False)
        include_input_dim = config.get('include_input_dim_in_label', False)
        ylimit = config.get('ylimit', 2.5)
        label_format = config.get('label_format', 'standard')

    def make_label(row):
        if label_format == 'compact':
            values = []
            batch_prefix = None
            for c in grp_cols:
                if c == 'batch_size':
                    if include_batch_size:
                        batch_val = str(int(row.get('original_batch_size', row.get('batch_size'))))
                        if double_row_batch_size:
                            batch_prefix = f"Batch Size = {batch_val}"
                        else:
                            values.append(batch_val)
                elif c == '#head':
                    values.append(str(int(row.get('original_head', row.get('#head')))))
                elif c == 'input_dim':
                    if include_input_dim or 'FC' in sheet_name:
                        values.append(str(int(row.get('input_dim'))))
                elif c == 'output_dim':
                    values.append(str(int(row.get('output_dim'))))
                elif c == 'vec_len':
                    values.append(str(int(row.get('vec_len'))))
            if 'FC' in sheet_name:
                values = [values[0], str(int(values[2]) * int(values[1])), values[3]]
            compact_label = 'x'.join(values)
            if batch_prefix:
                return f"{batch_prefix} {compact_label}"
            return compact_label
        else:
            label_map = {
                'batch_size': ('B', include_batch_size, row.get('original_batch_size', row.get('batch_size'))),
                '#head': ('H', True, row.get('original_head', row.get('#head'))),
                'input_dim': ('I', include_input_dim, row.get('input_dim')),
                'output_dim': ('O', True, row.get('output_dim')),
                'vec_len': ('V', True, row.get('vec_len'))
            }
            parts = []
            for c in grp_cols:
                if c in label_map:
                    prefix, should_include, value = label_map[c]
                    if should_include and value is not None:
                        parts.append(f"{prefix}{int(value)}")
                else:
                    parts.append(str(row[c]))
            return ' '.join(parts)
    
    plot_df['label'] = plot_df.apply(make_label, axis=1)

    if 'original_batch_size' in plot_df.columns and 'original_head' in plot_df.columns:
        sort_cols = ['original_batch_size', 'original_head']
        if 'input_dim' in plot_df.columns:
            sort_cols.append('input_dim')
        if 'output_dim' in plot_df.columns:
            sort_cols.append('output_dim')
        plot_df = plot_df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        sort_cols = [c for c in ['batch_size', '#head', 'input_dim', 'output_dim', 'vec_len'] if c in plot_df.columns]
        if sort_cols:
            plot_df = plot_df.sort_values(by=sort_cols).reset_index(drop=True)

    for idx, row in plot_df.iterrows():
        label_parts = []
        for c in grp_cols:
            if c == 'batch_size' and 'original_batch_size' in row:
                label_parts.append(f"batch_size={int(row['original_batch_size'])}")
            elif c == '#head' and 'original_head' in row:
                label_parts.append(f"#head={int(row['original_head'])}")
                label_parts.append(f"effective_#head={int(row[c])}")
            else:
                label_parts.append(f"{c}={row[c]}")

    labels = plot_df['label'].tolist()
    baseline_raw = plot_df['baseline(ms)'].astype(float).tolist()
    our_raw = plot_df['our_time(ms)'].astype(float).tolist()
    
    if has_gpu:
        gpu_raw = plot_df['GPU_time(ms)'].astype(float).tolist()
        baseline = [g / b if b > 0 else 0 for b, g in zip(baseline_raw, gpu_raw)]
        our = [g / o if o > 0 else 0 for o, g in zip(our_raw, gpu_raw)]
        gpu = [1.0] * len(gpu_raw)

        gpu_geomean = 1.0
        baseline_geomean = stats.hmean([x for x in baseline if x > 0]) if baseline else 0
        our_geomean = stats.hmean([x for x in our if x > 0]) if our else 0

        labels.append('Harmean')
        gpu.append(gpu_geomean)
        baseline.append(baseline_geomean)
        our.append(our_geomean)
    else:
        baseline = baseline_raw
        our = our_raw

    num_groups = len(labels)
    index = range(num_groups)

    rcParams.update({
        'font.family': "Ubuntu",
        'font.weight': "bold",
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.titlecolor': 'darkred',
        'figure.labelweight': 'bold',
        'figure.labelsize': 12,
        'legend.fontsize': 10,
        'axes.axisbelow': True
    })
    
    if 'Attacc' in sheet_name:
        baseline_name = 'AttAcc'
        our_name = 'AttAcc + DCC'
    elif 'HBMPIM' in sheet_name or 'HBM' in sheet_name:
        baseline_name = 'HBM-PIM'
        our_name = 'HBM-PIM + DCC'
    else:
        baseline_name = 'Baseline'
        our_name = 'DCC'
    
    cmap = matplotlib.colormaps.get_cmap('bone')
    
    if config_fig_width is not None:
        fig_width = config_fig_width
    else:
        fig_width = max(10, num_groups * 0.5)
    
    fig_height = config_fig_height
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        standalone_plot = True
    else:
        fig = ax.get_figure()
        standalone_plot = False
    
    if has_gpu:
        colors = (bar_colors[0], bar_colors[1])
        offsets = [-bar_width/2, bar_width/2]
        data_sets = [(baseline_name, baseline, colors[0]), (our_name, our, colors[1])]
        
        for (label, data, color), offset in zip(data_sets, offsets):
            plt.bar([x + offset for x in index], data, bar_width, label=label, 
                   edgecolor='black', linewidth=0.5, color=color)
            add_bar_text(ax, [x + offset for x in index], data, show_numbers, ylimit, bar_label_offset, bar_label_fontsize)
        ncol = 2
    else:
        colors = (bar_colors[0], bar_colors[1])
        offsets = [-bar_width/2, bar_width/2]
        data_sets = [(baseline_name, baseline, colors[0]), (our_name, our, colors[1])]
        
        for (label, data, color), offset in zip(data_sets, offsets):
            plt.bar([x + offset for x in index], data, bar_width, label=label, 
                   edgecolor='black', linewidth=0.5, color=color)
            add_bar_text(ax, [x + offset for x in index], data, show_numbers, ylimit, bar_label_offset, bar_label_fontsize)
        ncol = 2

    if has_gpu:
        if show_ylabel:
            ax.set_ylabel('Speedup over GPU', fontweight='bold', fontsize=ylabel_fontsize)
        ax.set_ylim(0, ylimit)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=1.0, zorder=2)
    else:
        if show_ylabel:
            ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=ylabel_fontsize)
        max_val = max(max(baseline), max(our))
        ax.set_ylim(0, max_val * 1.1)

    ha = 'center' if xlabel_rotation == 0 else 'right'
    ax.set_xticks([x for x in index])
    
    if double_row_batch_size:
        batch_labels = []
        other_labels = []
        for label in labels:
            parts = label.split()
            batch_part = None
            other_parts = []
            if len(parts) >= 3 and parts[0] == 'Batch' and parts[1] == 'Size':
                if parts[2] == '=' and len(parts) >= 4:
                    batch_part = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}"
                    other_parts = parts[4:]
                else:
                    batch_part = f"{parts[0]} {parts[1]} {parts[2]}"
                    other_parts = parts[3:]
            else:
                other_parts = parts

            batch_labels.append(batch_part if batch_part else '')
            other_labels.append(' '.join(other_parts))

        ax.set_xticklabels(other_labels, rotation=xlabel_rotation, ha=ha, fontsize=xlabel_fontsize)

        if batch_labels:
            batch_groups = []
            current_batch = batch_labels[0]
            start_idx = 0
            
            for idx in range(1, len(batch_labels)):
                if batch_labels[idx] != current_batch:
                    batch_groups.append((current_batch, start_idx, idx - 1))
                    current_batch = batch_labels[idx]
                    start_idx = idx

            batch_groups.append((current_batch, start_idx, len(batch_labels) - 1))

            for batch_label, start, end in batch_groups:
                if batch_label:
                    center_pos = (start + end) / 2.0
                    batch_y_center = (batch_label_y_position - 0.15) / 2.0
                    ax.text(center_pos, batch_y_center, batch_label, 
                           transform=ax.get_xaxis_transform(),
                           ha='center', va='center',
                           fontsize=xlabel_fontsize, fontweight='bold')

            if draw_batch_borders:
                from matplotlib.lines import Line2D
                trans = ax.get_xaxis_transform()

                table_left = -0.5
                table_right = len(batch_labels) - 0.5

                for idx in range(len(batch_labels) + 1):
                    x_pos = idx - 0.5
                    is_border = (idx == 0 or idx == len(batch_labels))
                    if is_border:
                        line = Line2D([x_pos, x_pos], [batch_label_y_position, 0], 
                                    transform=trans, color='black', linewidth=1, clip_on=False)
                        ax.add_line(line)
                    else:
                        line = Line2D([x_pos, x_pos], [batch_label_y_position * 0.5, 0], 
                                    transform=trans, color='black', linewidth=1, clip_on=False)
                        ax.add_line(line)

                line = Line2D([table_left, table_left], [-0.15, batch_label_y_position], 
                            transform=trans, color='black', linewidth=1, clip_on=False)
                ax.add_line(line)

                line = Line2D([table_right, table_right], [-0.15, batch_label_y_position], 
                            transform=trans, color='black', linewidth=1, clip_on=False)
                ax.add_line(line)

                for batch_label, start, end in batch_groups:
                    if start > 0:
                        line = Line2D([start - 0.5, start - 0.5], [-0.15, batch_label_y_position], 
                                    transform=trans, color='black', linewidth=1, clip_on=False)
                        ax.add_line(line)

                line_middle = Line2D([table_left, table_right], [batch_label_y_position, batch_label_y_position], 
                                transform=trans, color='black', linewidth=1, clip_on=False)
                ax.add_line(line_middle)

                line_top = Line2D([table_left, table_right], [0, 0], 
                                transform=trans, color='black', linewidth=1, clip_on=False)
                ax.add_line(line_top)
                line_bottom = Line2D([table_left, table_right], [-0.15, -0.15], 
                                   transform=trans, color='black', linewidth=1, clip_on=False)
                ax.add_line(line_bottom)
    else:
        ax.set_xticklabels(labels, rotation=xlabel_rotation, ha=ha, fontsize=xlabel_fontsize)
    
    if standalone_plot and not is_combined and show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=(1.0, 1.0), ncol=ncol, frameon=False, fontsize=legend_fontsize, labelspacing=0.1, handletextpad=0.2, columnspacing=0.5)
    
    ax.grid(axis='y')
    
    op_name_config = sheet_config.get('op_name', None)
    if op_name_config and standalone_plot:
        ax.text(
            op_name_config.get('x', 0.5),
            op_name_config.get('y', 0.85),
            op_name_config.get('text', ''),
            transform=ax.transAxes,
            fontsize=op_name_config.get('fontsize', 16),
            fontweight=op_name_config.get('fontweight', 'bold'),
            ha=op_name_config.get('ha', 'center'),
            va=op_name_config.get('va', 'top')
        )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(-0.5, num_groups - 0.5)
    
    if standalone_plot:
        plt.subplots_adjust(bottom=subplots_adjust_bottom)

    if save and standalone_plot:
        if has_gpu:
            import matplotlib.ticker as ticker
            current_ticks = ax.yaxis.get_major_locator().tick_values(0, ylimit)
            yticks = list(current_ticks)
            if 1.0 not in yticks:
                yticks.append(1.0)
                yticks = [t for t in yticks if 0 <= t <= ylimit]
                yticks.sort()
                ax.set_yticks(yticks)
        
        prefix = 'speedup' if has_gpu else 'latency'
        outname = os.path.join(plot_path, f'{prefix}_{sheet_name}.pdf')
        csv_outname = os.path.join(plot_path, f'{prefix}_{sheet_name}.csv')
        
        plt.savefig(outname, bbox_inches='tight')
        
        csv_data = plot_df.copy()
        if has_gpu:
            csv_data['GPU_speedup'] = gpu[:-1]
            csv_data[f'{baseline_name}_speedup'] = baseline[:-1]
            csv_data[f'{our_name}_speedup'] = our[:-1]
        
        csv_data.to_csv(csv_outname, index=False)
        
    
    return {
        'num_groups': num_groups,
        'has_gpu': has_gpu,
        'ncol': ncol,
        'legend_fontsize': legend_fontsize,
        'legend_height': legend_height,
        'show_legend': show_legend,
        'legend_loc': legend_loc
    }

if __name__ == '__main__':
    here = os.path.dirname(__file__)

    plot_path = "../figures/"
    config_path = os.path.join(here, 'kernels.json')
    
    filter_config = None
    horizontal_groups = []
    show_numbers = True
    bar_colors = ["#D3D3D3", "#696969"]
    available_sheets = []
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print(f"Loaded filter configuration from {config_path}")

        if 'global_options' in config_data:
            show_numbers = config_data['global_options'].get('show_numbers_on_bars', True)
            bar_colors = config_data['global_options'].get('bar_colors', ["#D3D3D3", "#696969"])

        filter_config = {k: v for k, v in config_data.items() 
                        if k not in ['global_options']}

        available_sheets = list(filter_config.keys())
        print(f"Found {len(available_sheets)} sheets in config: {available_sheets}")
    else:
        print(f"Warning: Filter configuration file not found at {config_path}")

    for sheet in available_sheets:
        if filter_config and sheet in filter_config:
            enabled = filter_config[sheet].get('enabled', True)
            if not enabled:
                print(f"\nSkipping sheet: {sheet} (disabled in config)")
                continue
        
        print(f"Plot for: {sheet}")
        plot_prediction_results_excel(plot_path, sheet_name=sheet, filter_config=filter_config, show_numbers=show_numbers, bar_colors=bar_colors)



