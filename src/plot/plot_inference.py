import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import json


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


def load_config(config_file='plot_config.json'):
  with open(config_file, 'r') as f:
    return json.load(f)


def load_and_filter_data(csv_file, models, lin_lout_pairs):
  df = pd.read_csv(csv_file)

  pair_conditions = []
  for lin, lout in lin_lout_pairs:
    pair_cond = (df['Lin'] == lin) & (df['Lout'] == lout)
    pair_conditions.append(pair_cond)

  combined_pair_cond = pair_conditions[0]
  for cond in pair_conditions[1:]:
    combined_pair_cond = combined_pair_cond | cond

  mask = df['model'].isin(models) & combined_pair_cond

  filtered_df = df[mask].copy()

  cols = ['name', 'model', 'Lin', 'Lout', "bs", 'g_time (ms)',
          'g_matmul', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g_etc']
  filtered_df = filtered_df[cols]

  return filtered_df


def normalize_to_gpu(df):
  normalized_data = []

  for (model, lin, lout), group in df.groupby(['model', 'Lin', 'Lout']):
    # Find GPU baseline
    gpu_row = group[group['name'] == 'GPU']
    if len(gpu_row) == 0:
      continue

    gpu_time = gpu_row['g_time (ms)'].values[0]

    for _, row in group.iterrows():
      norm_factor = row['g_time (ms)'] / gpu_time
      normalized_row = {
        'name': row['name'],
        'model': model,
        'Lin': lin,
        'Lout': lout,
        'normalized_time': row['g_time (ms)'] / gpu_time,
        'g_matmul_norm': row['g_matmul'] / gpu_time,
        'g_qkv_time_norm': row['g_qkv_time'] / gpu_time,
        'g_prj_time_norm': row['g_prj_time'] / gpu_time,
        'g_ff_time_norm': row['g_ff_time'] / gpu_time,
        'g_etc_norm': row['g_etc'] / gpu_time
      }
      normalized_data.append(normalized_row)

  return pd.DataFrame(normalized_data)


def plot_combined_figure(all_data, output_file, config):

  # Extract config values
  fig_cfg = config['figure']
  data_cfg = config['data']
  bar_cfg = config['bars']
  axes_cfg = config['axes']
  label_cfg = config['labels']
  sep_cfg = config['separators']
  legend_cfg = config['legend']
  layout_cfg = config['layout']

  models = data_cfg['models']
  model_labels = data_cfg['model_labels']
  lin_lout_pairs = data_cfg['lin_lout']
  batch_sizes = data_cfg['batchs']


  combined_width = fig_cfg['width'] * 2  # Double the width for 4 subplots
  fig, axes = plt.subplots(1, len(batch_sizes)*len(models), figsize=(combined_width, fig_cfg['height']), sharey=True)



  devices_display = ['GPU', 'AttAcc$_{\mathrm{Base}}$', 'AttAcc$_{\mathrm{Base}}$+DCC', 'AttAcc$_{\mathrm{Full}}$', 'AttAcc$_{\mathrm{Full}}$+DCC']
  devices_internal = data_cfg['devices']
  colors = bar_cfg['colors']
  patterns = bar_cfg.get('patterns', {})
  patterns = {}
  component_labels_map = bar_cfg['component_labels']

  for batch_idx, batch_size in enumerate(batch_sizes):

    for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
      ax_idx = batch_idx * len(models) + model_idx
      ax = axes[ax_idx]

      model_data = all_data[batch_size]
      model_data = model_data[model_data['model'] == model]

      num_configs = len(lin_lout_pairs)
      num_devices = 5

      bar_width = bar_cfg['width']
      group_spacing = bar_cfg['group_spacing']

      x_positions = []

      for i, (lin, lout) in enumerate(lin_lout_pairs):
        for j in range(num_devices):
          x_pos = i * (num_devices + group_spacing) + j
          x_positions.append(x_pos)

      # Plot stacked bars
      bottom_values = np.zeros(len(x_positions))

      components = ['g_matmul_norm', 'g_qkv_time_norm', 'g_prj_time_norm',
                    'g_ff_time_norm', 'g_etc_norm']
      component_labels = [component_labels_map[comp.replace('_norm', '')] for comp in components]

      # Build data matrix
      data_matrix = []
      for lin, lout in lin_lout_pairs:
        for device_name in devices_internal:
          row_data = model_data[(model_data['Lin'] == lin) &
                                (model_data['Lout'] == lout) &
                                (model_data['name'] == device_name)]
          if len(row_data) > 0:
            values = [row_data[comp].values[0] for comp in components]
          else:
            values = [0] * len(components)
          data_matrix.append(values)

      data_matrix = np.array(data_matrix).T

      for i, (comp, label) in enumerate(zip(components, component_labels)):
        color_key = comp.replace('_norm', '')
        color = colors[color_key]
        pattern = patterns.get(color_key, '')

        ax.bar(x_positions, data_matrix[i], bar_width,
               bottom=bottom_values, label=label if ax_idx == 0 else '', color=color,
               edgecolor=bar_cfg['edgecolor'], linewidth=bar_cfg['edgewidth'],
               hatch=pattern)
        bottom_values += data_matrix[i]

      if label_cfg.get('show_stack_values', False):
        stack_fontsize = label_cfg.get('stack_value_fontsize', 7)
        stack_fontweight = label_cfg.get('stack_value_fontweight', 'normal')
        stack_color = label_cfg.get('stack_value_color', 'black')
        min_height = label_cfg.get('stack_value_min_height', 0.05)

        stack_bottom = np.zeros(len(x_positions))

        for i, comp in enumerate(components):
          segment_heights = data_matrix[i]

          for j, (x_pos, height) in enumerate(zip(x_positions, segment_heights)):
            if height > min_height:
              y_pos = stack_bottom[j] + height / 2
              ax.text(x_pos, y_pos, f'{height:.3f}',
                      ha='center', va='center',
                      fontsize=stack_fontsize,
                      fontweight=stack_fontweight,
                      color=stack_color, rotation=0)

          stack_bottom += segment_heights

      speedup_fontsize = label_cfg.get('speedup_fontsize', 9)
      speedup_fontweight = label_cfg.get('speedup_fontweight', 'bold')
      speedup_color = label_cfg.get('speedup_color', 'darkgreen')
      speedup_offset = label_cfg.get('speedup_offset', 0.02)

      # Calculate speedup for each bar (skip GPU bars)
      for i, (lin, lout) in enumerate(lin_lout_pairs):
        for j, device_name in enumerate(devices_internal):
          if device_name == 'GPU':
            continue

          idx = i * num_devices + j
          x_pos = x_positions[idx]
          total_height = bottom_values[idx]

          row_data = model_data[(model_data['Lin'] == lin) &
                                (model_data['Lout'] == lout) &
                                (model_data['name'] == device_name)]

          if len(row_data) > 0:
            normalized_time = row_data['normalized_time'].values[0]
            speedup = 1.0 / normalized_time if normalized_time > 0 else 0

            # Position text above the bar
            y_pos = total_height + speedup_offset
            ax.text(x_pos, y_pos, f'{speedup:.2f}',
                    ha='center', va='bottom',
                    fontsize=speedup_fontsize,
                    fontweight=speedup_fontweight,
                    color=speedup_color, rotation=0)

      ax.set_xticks(x_positions)
      display_device_names = devices_display * num_configs
      ax.set_xticklabels(display_device_names, fontsize=label_cfg['device_fontsize'], rotation=90)

      group_centers = []
      for i in range(num_configs):
        start_idx = i * num_devices
        center_pos = np.mean(x_positions[start_idx:start_idx + num_devices])
        group_centers.append(center_pos)

      lin_lout_labels = [f'L$_{{in}}$:{lin},L$_{{out}}$:{lout}' for lin, lout in lin_lout_pairs]
      for center_x, label in zip(group_centers, lin_lout_labels):
        x_in_data = center_x
        ax.text(x_in_data, label_cfg['linlout_label_y'], label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=label_cfg['linlout_fontsize'])

      margin = bar_width * 0.9
      ax.set_xlim(x_positions[0] - margin, x_positions[-1] + margin)

      trans = ax.get_xaxis_transform()

      linlout_label_y = label_cfg['linlout_label_y']
      sep_end_y = sep_cfg['end_y']
      sep_color = sep_cfg['color']
      sep_linewidth = sep_cfg['linewidth']

      xlim = ax.get_xlim()
      x_min = xlim[0]
      x_max = xlim[1]

      line = Line2D([x_min, x_min], [0, sep_end_y],
                    transform=trans, color=sep_color, linewidth=sep_linewidth, clip_on=False)
      ax.add_line(line)

      line = Line2D([x_max, x_max], [0, sep_end_y],
                    transform=trans, color=sep_color, linewidth=sep_linewidth, clip_on=False)
      ax.add_line(line)

      for i in range(1, num_configs):
        last_bar_prev_group = x_positions[(i - 1) * num_devices + num_devices - 1]
        first_bar_curr_group = x_positions[i * num_devices]
        separator_x = (last_bar_prev_group + first_bar_curr_group) / 2
        line = Line2D([separator_x, separator_x], [0, sep_end_y],
                      transform=trans, color=sep_color, linewidth=sep_linewidth, clip_on=False)
        ax.add_line(line)

      if ax_idx == 0:
        ax.set_ylabel(axes_cfg['ylabel'], fontsize=axes_cfg['ylabel_fontsize'],
                      fontweight=axes_cfg['ylabel_fontweight'])

      ax.set_ylim(axes_cfg['ylim_min'], axes_cfg['ylim_max'])
      ax.set_yticks(np.arange(axes_cfg['ylim_min'], axes_cfg['ylim_max'] + 0.1, axes_cfg['ytick_step']))
      ax.grid(axis='y', alpha=axes_cfg['grid_alpha'], linestyle=axes_cfg['grid_linestyle'],
              linewidth=axes_cfg['grid_linewidth'])

      ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.8, zorder=2)

      # Add model label with batch size below x-axis (below Lin/Lout labels)
      title_text = f'{model_label} (Batch Size={batch_size})'
      ax.text(0.5, label_cfg['model_label_y'], title_text, transform=ax.transAxes,
              ha='center', fontsize=label_cfg['model_fontsize'],
              fontweight=label_cfg['model_fontweight'])

      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', ncol=legend_cfg['ncol'],
             bbox_to_anchor=tuple(legend_cfg['bbox_to_anchor']),
             frameon=legend_cfg['frameon'], fontsize=legend_cfg['fontsize'])

  plt.subplots_adjust(wspace=0.04)

  # Save figure
  plt.savefig(output_file, dpi=fig_cfg['dpi'], bbox_inches='tight')

  plt.close()

  base_filename = output_file.replace('.pdf', '').replace('.png', '')

  combined_csv = f"{base_filename}.csv"
  combined_df = pd.concat([all_data[ba] for ba in batch_sizes], ignore_index=True)
  combined_df.to_csv(combined_csv, index=False)
  print(f"Plot finished!  Total bars: {len(combined_df)}")


def main():
  # Load configuration
  here = os.path.dirname(__file__)
  config_path = os.path.join(here, 'inference.json')
  config = load_config(config_path)

  data_cfg = config['data']
  models = data_cfg['models']
  lin_lout_pairs = data_cfg['lin_lout']

  all_data = load_and_filter_data("../results/@inference_results.csv", models, lin_lout_pairs)
  batch_data = {}
  for batch in [1,4]:
    temp = all_data[all_data['bs'] == batch]
    batch_data[batch] = normalize_to_gpu(temp)

  output_file = '../figures/inference.pdf'
  plot_combined_figure(batch_data, output_file, config)


if __name__ == '__main__':
  main()
