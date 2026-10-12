import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_theme(style="whitegrid")
out_dir = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/docs_and_notes/charts"
report_file = "/home/eelis/Desktop/Diplomityö/peopleTrackingWithLidar/results/ablation_tuning/final_report.md"

def parse_data():
    with open(report_file, 'r') as f:
        lines = f.readlines()

    data = []
    current_model = None

    for line in lines:
        if line.startswith('## Model:'):
            current_model = line.split('## Model:')[1].strip()
        elif current_model and line.startswith('|') and 'Dist Thresh' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 10:
                try:
                    amota = float(parts[3].replace('*', '')) * 100  # Convert to percentage
                    hota = float(parts[4]) * 100 if parts[4] else 0.0
                    idf1 = float(parts[5]) * 100 if parts[5] else 0.0
                    refl = float(parts[2].replace('x', ''))
                    data.append({'Model': current_model, 'Refl Weight': refl, 'AMOTA': amota, 'HOTA': hota, 'IDF1': idf1})
                except ValueError:
                    pass
    
    return pd.DataFrame(data)

def plot_sensitivity():
    df = parse_data()
    # Calculate means grouping by Model and Refl Weight
    means = df.groupby(['Model', 'Refl Weight']).mean().reset_index()
    
    # Filter for the two models we want to compare
    target_models = ['cp_decoupled', 'hmfn_swap']
    plot_df = means[means['Model'].isin(target_models)].copy()
    
    # Rename models for presentation
    model_names = {
        'cp_decoupled': 'CenterPoint Decoupled (Native 3D)',
        'hmfn_swap': 'HMFN Swap (Forced Physics Equation)'
    }
    plot_df['Model'] = plot_df['Model'].map(model_names)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    metrics = ['AMOTA', 'HOTA', 'IDF1']
    titles = ['Detection Volume (AMOTA %)', 'Tracking Coherence (HOTA %)', 'Identity Preservation (IDF1 %)']
    
    colors = {'CenterPoint Decoupled (Native 3D)': '#1f77b4', 'HMFN Swap (Forced Physics Equation)': '#d62728'}
    markers = {'CenterPoint Decoupled (Native 3D)': 'o', 'HMFN Swap (Forced Physics Equation)': 's'}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model in model_names.values():
            model_data = plot_df[plot_df['Model'] == model]
            ax.plot(model_data['Refl Weight'], model_data[metric], 
                    marker=markers[model], color=colors[model], linewidth=2.5, markersize=8, label=model)
        
        ax.set_title(titles[i], fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Reflectivity Tracker Penalty (x)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Mean Score (%)', fontsize=12)
        ax.set_xticks([0.0, 0.1, 0.3, 0.5])
        
        # Add a subtle grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=11, frameon=True)
    
    plt.suptitle('Tracking Metric Sensitivity to Reflectivity Penalty:\nNative 3D Feature Decoupling vs Forced Physics', 
                 fontsize=15, fontweight='bold', y=1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '6_metric_sensitivity_analysis.pdf'), bbox_inches='tight')
    print("Saved sensitivity chart to", os.path.join(out_dir, '6_metric_sensitivity_analysis.pdf'))

if __name__ == '__main__':
    plot_sensitivity()
