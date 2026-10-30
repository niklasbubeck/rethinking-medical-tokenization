import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, df, output_dir, style='darkgrid'):
        self.df = df
        self.output_dir = output_dir
        sns.set_style(style)
        os.makedirs(output_dir, exist_ok=True)

    def save_plot(self, fig, filename):
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_numerical(self, prop, metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in metrics:
            ax.plot(self.df[prop], self.df[metric], marker='o', label=metric.upper())
        ax.set_title(f'{prop} vs Metrics')
        ax.set_xlabel(prop)
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True)
        self.save_plot(fig, f'{prop}_vs_metrics.png')

    def plot_categorical(self, prop, metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        melted = pd.melt(self.df, id_vars=[prop], value_vars=metrics,
                         var_name='Metric', value_name='Value')
        sns.boxplot(data=melted, x=prop, y='Value', hue='Metric', ax=ax)
        ax.set_title(f'{prop} vs Metrics')
        ax.set_xlabel(prop)
        ax.set_ylabel('Metric Value')
        plt.xticks(rotation=45)
        ax.legend(title='Metric')
        self.save_plot(fig, f'{prop}_vs_metrics.png')

    def plot_scatter_matrix(self, properties, metrics):
        fig = sns.pairplot(self.df, vars=properties + metrics,
                           hue='task_name' if 'task_name' in self.df.columns else None)
        fig.fig.suptitle('Scatter Matrix of Properties and Metrics', y=1.02)
        self.save_plot(fig.fig, 'scatter_matrix.png')

    def plot_correlation_heatmap(self, properties, metrics):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self.df[properties + metrics].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Heatmap')
        self.save_plot(fig, 'correlation_heatmap.png')

    def create_all_plots(self, properties, metrics, advanced=False):
        for prop in properties:
            if self.df[prop].dtype == 'object' or self.df[prop].nunique() < 10:
                self.plot_categorical(prop, metrics)
            else:
                self.plot_numerical(prop, metrics)

        if advanced:
            self.plot_scatter_matrix(properties, metrics)
            self.plot_correlation_heatmap(properties, metrics)