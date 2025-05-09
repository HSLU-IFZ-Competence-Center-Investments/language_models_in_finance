import pandas as pd
from dataclasses import dataclass
from typing import List
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from math import ceil
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties




file_path = './results/experiment_results.csv'
model_names = [
    "TinyLlama-v1.1",
    "Phi-4-mini-instruct",
    "gemma-3-4b-it",
    "zephyr-7b-beta",
    "Mistral-7B-Instruct-v0.3",
    "Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Qwen-14B",
    "Mixtral-8x22B-Instruct-v0.1",
    "gpt-4.1"
]

custom_colors = [
    '#77C5D8', '#adca2a', '#ee6a87', '#fcc300',
    "#206A8A", "#386939", "#9F304D", "#D47F1C",
    "#459DC2", "#7D9E38", "#BB365D", "#F7A315",
    "#BAE0EA", "#D6E28C", "#F6A9B2", "#FFE086",
    "#DAEEF3", "#E9F0C1", "#FBDDD3", "#FFF0BE",
    ]

# === FONT ===
font_path = './fonts/FSALBERT.OTF'

# Check if font file exists
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found: {font_path}")

# Load font and get its name
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

@dataclass
class EvalResult:
    Date: str
    Model_name: str
    Data_scope: str
    RAG: bool
    TP: int
    FP: int
    TN: int
    FN: int
    unclear_res_rd: int
    unclear_res_ad: int
    Precision: float
    Recall: float
    Accuracy: float
    F1_Score: float
    Execution_time: float
    GPU_name: str

def load_eval_results(file_path: str) -> List[EvalResult]:
    results = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result = EvalResult(
                Date=row['Date'],
                Model_name=row['Model name'],
                Data_scope=row['Data scope'],
                RAG=row['RAG'].strip().lower() in ['true', '1', 'yes'],
                TP=int(row['TP']),
                FP=int(row['FP']),
                TN=int(row['TN']),
                FN=int(row['FN']),
                unclear_res_rd=int(row['Unclear result in real dataset']),
                unclear_res_ad=int(row['Unclear result in artificial dataset']),
                Precision=float(row['Precision']),
                Recall=float(row['Recall']),
                Accuracy=float(row['Accuracy']),
                F1_Score=float(row['F1 Score']),
                Execution_time=float(row['Execution time']),
                GPU_name=row['GPU name']
            )
            results.append(result)
    return results

# Radar chart plotting function with multiple scopes
def plot_combined_radar_chart(model_name, data_by_scope, output_folder):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for idx, (scope, metrics) in enumerate(data_by_scope.items()):
        values = [metrics[label] for label in labels] + [metrics[labels[0]]]
        ax.plot(angles, values, 'o-', label=scope, linewidth=2, color=custom_colors[idx % len(custom_colors)])
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} with RAG", size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    filename = f"{clean_filename(model_name)}_combined_RAG.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def clean_filename(text):
    return text.replace(' ', '_').replace('/', '-').replace('\\', '-')

def plot_rag_combined(df):
    # Filter to only RAG=True
    df_rag = df[df['RAG'] == True]

    # Data scopes to include
    target_scopes = ['International', 'Switzerland', 'Switzerland Nebenwerte']

    # Create output folder
    output_folder = './plots_rag_combined'
    os.makedirs(output_folder, exist_ok=True)

    for model_name, group in df_rag.groupby('Model_name'):
        data_by_scope = {}
        for scope in target_scopes:
            scope_group = group[group['Data_scope'] == scope]
            if not scope_group.empty:
                data_by_scope[scope] = {
                    'Accuracy': scope_group['Accuracy'].mean(),
                    'Precision': scope_group['Precision'].mean(),
                    'Recall': scope_group['Recall'].mean(),
                    'F1 Score': scope_group['F1_Score'].mean()
                }

        if data_by_scope:  # Only plot if at least one scope exists
            plot_combined_radar_chart(model_name, data_by_scope, output_folder)

def plot_compare_f1(df):
    # Filter for the 3 Data Scopes
    scopes = ['International', 'Switzerland', 'Switzerland Nebenwerte']
    df_scopes = df[df['Data_scope'].isin(scopes)]
    font_prop = font_manager.FontProperties(size=14)  # or whatever size you want


    # Group by Model_name, Data_scope, and RAG, then calculate mean F1 Score
    f1_scores_all_scopes = df_scopes.groupby(['Model_name', 'Data_scope', 'RAG'])['F1_Score'].mean().reset_index()

    # Plot comparison for each data scope
    for scope in scopes:
        # Filter data for the current scope
        f1_scope_data = f1_scores_all_scopes[f1_scores_all_scopes['Data_scope'] == scope]

        # Create a bar plot for the scope with the custom RAG color palette
        plt.figure(figsize=(12, 6))
        sns.barplot(data=f1_scope_data, x='Model_name', y='F1_Score', hue='RAG', palette=custom_colors[:2],order=model_names)

        # Set y-axis limit for consistent scale
        plt.ylim(0.0, 1.0)

        # Customize plot with font
        # plt.title(f'Comparison of F1 Score by Model and RAG Status ({scope} Scope)', fontsize=16, fontproperties=font_prop)
        plt.xlabel('Model Name', fontproperties=font_prop)
        plt.ylabel('F1 Score', fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right')

        # Apply font to x-axis tick labels
        for label in plt.gca().get_xticklabels():
            label.set_fontproperties(font_prop)

        # Apply font to legend
        legend = plt.legend(title='RAG', loc='upper right')
        legend.set_title('RAG', prop=font_prop)
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'./plots/plots_f1_comparison_{scope.lower().replace(" ", "_")}_scope.png', dpi=300)
        plt.close()

def plot_runtime(df):
    # Filter only RAG = True
    df_rag_true = df[df['RAG'] == True]
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    # Scatter plot: color by Model, shape by Scope
    sns.scatterplot(
        data=df_rag_true,
        x='F1_Score',
        y='Execution_time',
        hue='Model_name',
        style='Data_scope',
        s=100,
        palette=custom_colors,
        hue_order=model_names
    )
    font_prop_legend = FontProperties(fname=font_path, size=14)  
    # plt.title('Execution Time vs F1 Score (RAG = True)', fontsize=16, fontproperties=font_prop)
    plt.xlabel('F1 Score', fontproperties=font_prop_legend)
    plt.ylabel('Execution Time (s)', fontproperties=font_prop_legend)
    plt.yscale('log')
    # Apply font to legend title and labels
    legend = plt.legend(
        title='Model / Scope',
        bbox_to_anchor=(0.5, -0.15),  
        loc='upper center',
        ncol=3,  
    )
    
    legend.set_title('Model / Scope', prop=font_prop_legend)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop_legend)

    plt.tight_layout()
    plt.savefig('./plots/execution_time_vs_f1_rag_true_by_model_color.png', dpi=300)
    plt.close()


def heatmap(data, include_rag=True):
    df = pd.DataFrame([r.__dict__ for r in data])

    if include_rag:
        group_cols = ['Model_name', 'RAG']
    df = df[df['RAG'] == True]
        
    df_agg = df.groupby(group_cols)[['Precision', 'Recall', 'Accuracy', 'F1_Score']].mean().reset_index()

    df_agg = df_agg.set_index('Model_name')
    metric_df = df_agg[['Precision', 'Recall', 'Accuracy', 'F1_Score']]

        # Your custom color palette
    custom_colors = ['#f0f0f0','#daeef3', '#bae0ea', '#77C5D8', '#449dc2', '#206a8a']
    
    # Create a custom colormap using your colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

    plt.figure(figsize=(10, 6))
    sns.heatmap(metric_df, annot=True, cmap=custom_cmap, vmin=0.0, vmax=1.0, linewidths=0.5)
    plt.title('Model Evaluation Metrics Heatmap')
    plt.ylabel('Model')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.savefig('./plots/heatmap.png', dpi=300)


def plot_fp_comparison(data, isRag):
    # Convert EvalResult list to DataFrame
    df = pd.DataFrame([r.__dict__ for r in data])
    df = df[df['Data_scope'] == 'International']


    # Filter data based on the isRag flag
    if isRag:
        df_filtered = df[df['RAG'] == True]
        rag_status = "RAG = True"
    else:
        df_filtered = df[df['RAG'] == False]
        rag_status = "RAG = False"

    # Group by Model_name and sum False Positives (FP)
    fp_comparison = df_filtered.groupby('Model_name')['FP'].sum().reset_index()

    # Custom color palette
    custom_colors = ['#77C5D8', '#adca2a', '#ee6a87', '#fcc300', '#206a8a', '#386939', '#9F304d']
    
    # Ensure custom colors are applied to models in order
    models = fp_comparison['Model_name'].tolist()
    color_map = {model: custom_colors[i % len(custom_colors)] for i, model in enumerate(models)}

    # Map the colors to the barplot
    colors = [color_map[model] for model in models]

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=fp_comparison, x='Model_name', y='FP', hue='Model_name', palette=color_map, legend=False)

    plt.title(f'Comparison of False Positives ({rag_status})', fontsize=16)
    plt.xlabel('Model Name', fontsize=14)
    plt.ylabel('False Positives', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'./plots/fp_comparison_{rag_status.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()


def plot_combined_radar_chart(ax, model_name, data_by_scope):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    colors = ['#77C5D8', '#ee6a87', '#adca2a']
    for i, (scope, metrics) in enumerate(data_by_scope.items()):
        values = [metrics[label] for label in labels] + [metrics[labels[0]]]
        ax.plot(angles, values, 'o-', label=scope, linewidth=2, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(model_name, size=10)


def plot_combined_radar_chart(ax, model_name, data_by_scope, colors, font_prop):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    for i, (scope, metrics) in enumerate(data_by_scope.items()):
        values = [metrics[label] for label in labels] + [metrics[labels[0]]]
        ax.plot(angles, values, 'o-', label=scope, linewidth=2, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=font_prop)  # Set radar axis labels
    ax.set_ylim(0, 1)
    ax.set_title(model_name, size=18, fontproperties=font_prop)  # Set title with font

def plot_rag_combined_new(df):
    df_rag = df[df['RAG'] == True]
    target_scopes = ['International', 'Switzerland', 'Switzerland Nebenwerte']
    font_prop = font_manager.FontProperties(size=18)  # or whatever size you want

    output_folder = './plots_rag_combined'
    os.makedirs(output_folder, exist_ok=True)

    n_models = len(model_names)
    n_cols = 3
    n_rows = ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 7), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    colors = [ '#77C5D8', '#adca2a', '#ee6a87']

    for i, model_name in enumerate(model_names):
        group = df_rag[df_rag['Model_name'] == model_name]
        data_by_scope = {}
        for scope in target_scopes:
            scope_group = group[group['Data_scope'] == scope]
            if not scope_group.empty:
                data_by_scope[scope] = {
                    'Accuracy': scope_group['Accuracy'].mean(),
                    'Precision': scope_group['Precision'].mean(),
                    'Recall': scope_group['Recall'].mean(),
                    'F1 Score': scope_group['F1_Score'].mean()
                }

        if data_by_scope:
            plot_combined_radar_chart(axes[i], model_name, data_by_scope, colors, font_prop)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),  # horizontally centered, slightly above top
        ncol=len(labels),           # all legend items in a single row
        title='Data Scope',
    )    
    legend.set_title('Data Scope', prop=font_prop)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)

    plt.savefig(os.path.join(output_folder, 'combined_radar_plots.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved: combined_radar_plots.png")


def plot_fn_comparison(data, isRag):
    # Convert EvalResult list to DataFrame
    df = pd.DataFrame([r.__dict__ for r in data])
    df = df[df['Data_scope'] == 'International']
    font_prop = font_manager.FontProperties(size=14)  # or whatever size you want

    # Filter data based on the isRag flag
    if isRag:
        df_filtered = df[df['RAG'] == True]
        rag_status = "RAG = True"
    else:
        df_filtered = df[df['RAG'] == False]
        rag_status = "RAG = False"

    # Group by Model_name and sum False Negatives (FN)
    fp_comparison = df_filtered.groupby('Model_name')['FN'].sum().reset_index()

    # Custom color palette
    custom_colors = ['#77C5D8', '#adca2a', '#ee6a87', '#fcc300', '#206a8a', '#386939', '#9F304d']
    models = fp_comparison['Model_name'].tolist()
    color_map = {model: custom_colors[i % len(custom_colors)] for i, model in enumerate(models)}
    colors = [color_map[model] for model in models]

    # Plotting
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=fp_comparison, x='Model_name', y='FN', hue='Model_name', palette=color_map, legend=False, order=model_names)

    # plt.title(f'Comparison of False Negatives ({rag_status})', fontsize=16, fontproperties=font_prop)
    plt.xlabel('Model Name', fontsize=14, fontproperties=font_prop)
    plt.ylabel('False Negatives', fontsize=14, fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    plt.ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontproperties=font_prop)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'./plots/fn_comparison_{rag_status.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    data = load_eval_results(file_path)
    # Convert dataclass list to DataFrame
    df = pd.DataFrame([vars(r) for r in data])

    # we re not using these plots
    # heatmap(data)
    # plot_fp_comparison(data, True)


    # plot_rag_combined_new(df)
    plot_compare_f1(df)
    # plot_runtime(df)
    # plot_fn_comparison(data, True)
