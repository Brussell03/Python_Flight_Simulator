import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_mc_results(results, mc_cfg, job_dir, job_name):
    """
    Converts raw multiprocessing outputs to a DataFrame, exports to CSV, 
    and dispatches requested statistical plots.
    """
    # 1. Aggregation and Export
    df = pd.DataFrame(results)
    
    out_dir = os.path.join(job_dir, f'output/{job_name}/monte_carlo')
    os.makedirs(out_dir, exist_ok=True)
    
    if mc_cfg.get('save_csv', True):
        csv_path = os.path.join(out_dir, f"{job_name}_mc_kpi.csv")
        df.to_csv(csv_path, index=False)
        print(f"Monte Carlo KPI data saved to: {csv_path}")

    # Isolate successful runs for plotting to prevent NaN math errors
    df_success = df[df['status'] == 'SUCCESS'].copy()
    
    if df_success.empty:
        print("No successful runs available for plotting.")
        return

    # 2. Config-Driven Plotting Dispatcher
    plot_cfg = mc_cfg.get('plots', {})
    if not plot_cfg:
        return

    # Set professional styling via Seaborn
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "xtick.bottom": True, "ytick.left": True})
    
    for plot_id, params in plot_cfg.items():
        p_type = params.get('type')
        
        if p_type == "1d_distribution":
            _plot_1d_distribution(df_success, params, out_dir, plot_id)
        elif p_type == "2d_density":
            _plot_2d_density(df_success, params, out_dir, plot_id)
        else:
            print(f"Warning: Unknown Monte Carlo plot type '{p_type}' requested for '{plot_id}'")

    # Suppress display block to allow the master script to continue to the next config
    plt.close('all')


def _plot_1d_distribution(df, params, out_dir, plot_id):
    """Generates a combined Histogram + Kernel Density Estimate (KDE) plot."""
    var_name = params.get('variable')
    if var_name not in df.columns:
        print(f"Error: Variable '{var_name}' not found in KPI results.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot histogram with a KDE overlay to show the continuous probability density
    sns.histplot(data=df, x=var_name, bins=params.get('bins', 30), kde=True, color="steelblue")
    
    # Calculate and plot statistical markers (Mean and 3-Sigma bounds)
    mean_val = df[var_name].mean()
    std_val = df[var_name].std()
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(mean_val + 3*std_val, color='orange', linestyle=':', linewidth=2, label=r'$+3\sigma$')
    plt.axvline(mean_val - 3*std_val, color='orange', linestyle=':', linewidth=2, label=r'$-3\sigma$')

    plt.title(params.get('title', f"Distribution of {var_name}"), fontsize=14, fontweight='bold')
    plt.xlabel(var_name, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"mc_{plot_id}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")


def _plot_2d_density(df, params, out_dir, plot_id):
    """Generates a scatter footprint with bivariate KDE density contours."""
    x_var = params.get('x')
    y_var = params.get('y')
    
    if x_var not in df.columns or y_var not in df.columns:
        print(f"Error: Variables '{x_var}' or '{y_var}' not found in KPI results.")
        return

    # JointGrid provides the central 2D plot with 1D marginal distributions on the axes
    g = sns.JointGrid(data=df, x=x_var, y=y_var, height=8, marginal_ticks=True)
    
    # Plot the core bivariate density contours
    g.plot_joint(sns.kdeplot, fill=True, cmap="mako", alpha=0.8, thresh=0.05, levels=8)
    
    # Overlay the exact scatter points for outlier visibility
    g.plot_joint(sns.scatterplot, color="black", s=10, alpha=0.5)
    
    # Plot the marginal distributions on the top and right axes
    g.plot_marginals(sns.histplot, color="steelblue", bins=40, kde=True)
    
    g.figure.suptitle(params.get('title', f"Bivariate Density: {y_var} vs {x_var}"), y=1.02, fontsize=14, fontweight='bold')
    
    save_path = os.path.join(out_dir, f"mc_{plot_id}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot: {save_path}")