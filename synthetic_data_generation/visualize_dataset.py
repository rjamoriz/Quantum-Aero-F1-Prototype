"""
Visualization and analysis tools for synthetic aerodynamic dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import List, Dict

from batch_orchestrator import DatasetStorage


class DatasetVisualizer:
    """Visualize and analyze synthetic aerodynamic dataset."""
    
    def __init__(self, dataset_dir: str):
        """Initialize visualizer."""
        self.storage = DatasetStorage(dataset_dir)
        self.scalars = self.storage.load_scalars()
        self.output_dir = Path(dataset_dir) / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def extract_data(self) -> Dict[str, np.ndarray]:
        """Extract data arrays from scalars."""
        data = {
            'CL': [],
            'CD': [],
            'L_over_D': [],
            'balance': [],
            'downforce_front': [],
            'downforce_rear': [],
            'main_plane_angle': [],
            'rear_wing_angle': [],
            'floor_gap': [],
            'DRS_open': [],
            'V_inf': [],
        }
        
        for sample in self.scalars:
            data['CL'].append(sample['global_outputs']['CL'])
            data['CD'].append(sample['global_outputs']['CD_total'])
            data['L_over_D'].append(sample['global_outputs']['L_over_D'])
            data['balance'].append(sample['global_outputs']['balance'])
            data['downforce_front'].append(sample['global_outputs']['downforce_front'])
            data['downforce_rear'].append(sample['global_outputs']['downforce_rear'])
            data['main_plane_angle'].append(sample['geometry_params']['main_plane_angle_deg'])
            data['rear_wing_angle'].append(sample['geometry_params']['rear_wing_angle_deg'])
            data['floor_gap'].append(sample['geometry_params']['floor_gap'])
            data['DRS_open'].append(float(sample['geometry_params']['DRS_open']))
            data['V_inf'].append(sample['flow_conditions']['V_inf'])
        
        return {k: np.array(v) for k, v in data.items()}
    
    def plot_efficiency_map(self, data: Dict[str, np.ndarray]):
        """Plot L/D vs CL efficiency map."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            data['CL'], 
            data['L_over_D'],
            c=data['CD'],
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        ax.set_xlabel('Lift Coefficient (CL)', fontsize=12)
        ax.set_ylabel('Aerodynamic Efficiency (L/D)', fontsize=12)
        ax.set_title('F1 Aerodynamic Efficiency Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Drag Coefficient (CD)', fontsize=11)
        
        # Add statistics
        textstr = f'Mean L/D: {data["L_over_D"].mean():.2f}\n'
        textstr += f'Max L/D: {data["L_over_D"].max():.2f}\n'
        textstr += f'n = {len(data["CL"])}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_map.png', dpi=300)
        print(f"✓ Saved: efficiency_map.png")
        plt.close()
    
    def plot_drag_polar(self, data: Dict[str, np.ndarray]):
        """Plot drag polar (CL vs CD)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by DRS state
        drs_open = data['DRS_open'] > 0.5
        
        ax.scatter(data['CD'][~drs_open], data['CL'][~drs_open], 
                  label='DRS Closed', alpha=0.6, s=50, c='blue')
        ax.scatter(data['CD'][drs_open], data['CL'][drs_open], 
                  label='DRS Open', alpha=0.6, s=50, c='red')
        
        ax.set_xlabel('Drag Coefficient (CD)', fontsize=12)
        ax.set_ylabel('Lift Coefficient (CL)', fontsize=12)
        ax.set_title('F1 Drag Polar', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'drag_polar.png', dpi=300)
        print(f"✓ Saved: drag_polar.png")
        plt.close()
    
    def plot_balance_distribution(self, data: Dict[str, np.ndarray]):
        """Plot downforce balance distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Balance histogram
        ax1.hist(data['balance'] * 100, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(data['balance'].mean() * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {data["balance"].mean()*100:.1f}%')
        ax1.set_xlabel('Front Downforce Balance (%)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Downforce Balance Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Front vs Rear downforce
        ax2.scatter(data['downforce_front'], data['downforce_rear'], 
                   alpha=0.6, s=50, c='steelblue')
        ax2.set_xlabel('Front Downforce (N)', fontsize=12)
        ax2.set_ylabel('Rear Downforce (N)', fontsize=12)
        ax2.set_title('Front vs Rear Downforce', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add 40% balance line
        max_df = max(data['downforce_front'].max(), data['downforce_rear'].max())
        x_line = np.linspace(0, max_df, 100)
        y_line = x_line * (0.6 / 0.4)  # 40% front = 60% rear
        ax2.plot(x_line, y_line, 'r--', alpha=0.5, label='40% balance')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'balance_distribution.png', dpi=300)
        print(f"✓ Saved: balance_distribution.png")
        plt.close()
    
    def plot_parameter_sensitivity(self, data: Dict[str, np.ndarray]):
        """Plot sensitivity of L/D to key parameters."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        params = [
            ('main_plane_angle', 'Main Plane Angle (deg)'),
            ('rear_wing_angle', 'Rear Wing Angle (deg)'),
            ('floor_gap', 'Floor Gap (mm)'),
            ('V_inf', 'Velocity (m/s)')
        ]
        
        for ax, (param, label) in zip(axes.flat, params):
            ax.scatter(data[param], data['L_over_D'], alpha=0.5, s=30)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('L/D', fontsize=11)
            ax.set_title(f'L/D vs {label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data[param], data['L_over_D'], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(data[param].min(), data[param].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.7, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_sensitivity.png', dpi=300)
        print(f"✓ Saved: parameter_sensitivity.png")
        plt.close()
    
    def plot_correlation_matrix(self, data: Dict[str, np.ndarray]):
        """Plot correlation matrix of key variables."""
        # Select variables for correlation
        vars_to_correlate = [
            'CL', 'CD', 'L_over_D', 'balance',
            'main_plane_angle', 'rear_wing_angle', 'floor_gap'
        ]
        
        # Create dataframe-like structure
        corr_data = np.array([data[v] for v in vars_to_correlate]).T
        corr_matrix = np.corrcoef(corr_data, rowvar=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(vars_to_correlate)))
        ax.set_yticks(np.arange(len(vars_to_correlate)))
        ax.set_xticklabels(vars_to_correlate, rotation=45, ha='right')
        ax.set_yticklabels(vars_to_correlate)
        
        # Add correlation values
        for i in range(len(vars_to_correlate)):
            for j in range(len(vars_to_correlate)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300)
        print(f"✓ Saved: correlation_matrix.png")
        plt.close()
    
    def generate_summary_report(self, data: Dict[str, np.ndarray]):
        """Generate text summary report."""
        report = []
        report.append("=" * 60)
        report.append("SYNTHETIC AERODYNAMIC DATASET SUMMARY")
        report.append("=" * 60)
        report.append(f"\nTotal Samples: {len(data['CL'])}")
        
        report.append("\n--- Aerodynamic Coefficients ---")
        for var in ['CL', 'CD', 'L_over_D']:
            report.append(f"{var:12s}: {data[var].mean():6.3f} ± {data[var].std():6.3f}  "
                         f"[{data[var].min():6.3f}, {data[var].max():6.3f}]")
        
        report.append("\n--- Downforce ---")
        total_df = data['downforce_front'] + data['downforce_rear']
        report.append(f"{'Total':12s}: {total_df.mean():8.1f} ± {total_df.std():8.1f} N")
        report.append(f"{'Front':12s}: {data['downforce_front'].mean():8.1f} ± {data['downforce_front'].std():8.1f} N")
        report.append(f"{'Rear':12s}: {data['downforce_rear'].mean():8.1f} ± {data['downforce_rear'].std():8.1f} N")
        report.append(f"{'Balance':12s}: {data['balance'].mean()*100:6.1f}% ± {data['balance'].std()*100:6.1f}% front")
        
        report.append("\n--- Geometry Parameters ---")
        for var in ['main_plane_angle', 'rear_wing_angle', 'floor_gap']:
            report.append(f"{var:20s}: {data[var].mean():6.2f} ± {data[var].std():6.2f}")
        
        report.append("\n--- Flow Conditions ---")
        report.append(f"{'Velocity':20s}: {data['V_inf'].mean():6.2f} ± {data['V_inf'].std():6.2f} m/s")
        report.append(f"{'DRS Open':20s}: {data['DRS_open'].sum():.0f} / {len(data['DRS_open'])} samples")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Saved: summary_report.txt")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60 + "\n")
        
        if not self.scalars:
            print("Error: No data found in dataset")
            return
        
        print(f"Loading {len(self.scalars)} samples...")
        data = self.extract_data()
        
        print("\nGenerating plots...")
        self.plot_efficiency_map(data)
        self.plot_drag_polar(data)
        self.plot_balance_distribution(data)
        self.plot_parameter_sensitivity(data)
        self.plot_correlation_matrix(data)
        
        print("\nGenerating summary report...")
        self.generate_summary_report(data)
        
        print("\n" + "=" * 60)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize synthetic aerodynamic dataset")
    parser.add_argument('--input', type=str, required=True, help='Dataset directory')
    
    args = parser.parse_args()
    
    visualizer = DatasetVisualizer(args.input)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    # Test with example dataset
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python visualize_dataset.py --input <dataset_directory>")
        print("\nExample: python visualize_dataset.py --input ./synthetic_dataset")
