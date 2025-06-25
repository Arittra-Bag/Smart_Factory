import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def test_plot_defect_rate(date_str="2025-06-25"):
    """Test the plotting function with the actual CSV data"""
    safety_check_records = "safety_check_records.csv"
    
    try:
        print(f"Testing plot for date: {date_str}")
        
        if not os.path.exists(safety_check_records):
            print("❌ No safety check records found")
            return None

        with open(safety_check_records, "r") as f:
            lines = f.readlines()
        
        print(f"Total lines in file: {len(lines)}")
        print("File contents:")
        for i, line in enumerate(lines):
            print(f"  {i}: {line.strip()}")

        # Find the section for the given date
        section_start = None
        section_end = None
        for i, line in enumerate(lines):
            if line.strip() == f"---- {date_str} ----":
                section_start = i
                print(f"Found section start at line {i}")
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("----"):
                        section_end = j
                        print(f"Found section end at line {j}")
                        break
                if section_end is None:
                    section_end = len(lines)
                    print(f"Section ends at end of file (line {len(lines)})")
                break

        if section_start is None:
            print(f"❌ No data found for date {date_str}")
            return None

        section_lines = lines[section_start+1:section_end]
        print(f"Section lines: {section_lines}")
        
        batch_sizes, defect_rates, timestamps = [], [], []
        for line in section_lines:
            line = line.strip()
            print(f"Processing line: '{line}'")
            if line == "" or line.startswith("Batch Size"):
                print(f"  Skipping line: '{line}'")
                continue
            parts = line.split(',')
            print(f"  Parts: {parts}")
            if len(parts) >= 3:
                try:
                    batch_size = int(float(parts[0]))
                    defect_rate = float(parts[1])
                    batch_sizes.append(batch_size)
                    defect_rates.append(defect_rate)
                    print(f"  Added: batch_size={batch_size}, defect_rate={defect_rate}")
                except Exception as e:
                    print(f"⚠️ Skipping malformed line: {line} - Error: {e}")
                    continue

        print(f"Final data: batch_sizes={batch_sizes}, defect_rates={defect_rates}")

        if not batch_sizes:
            print(f"❌ No valid data available for {date_str}")
            return None

        # Plotting
        print("Creating plot...")
        fig, ax = plt.subplots(figsize=(12, 8))
        sorted_data = sorted(zip(batch_sizes, defect_rates))
        
        print(f"Sorted data: {sorted_data}")
        
        # Check if we have data to plot
        if not sorted_data:
            print(f"❌ No valid data to plot for {date_str}")
            return None
            
        sorted_batch_sizes, sorted_defect_rates = zip(*sorted_data)
        print(f"Unpacked data: batch_sizes={sorted_batch_sizes}, defect_rates={sorted_defect_rates}")
        
        ax.plot(sorted_batch_sizes, sorted_defect_rates, marker='o', linewidth=2, markersize=8,
                color='#2ecc71', markeredgecolor='white', markeredgewidth=2, label='Defect Rate', alpha=0.7)

        # Mean line
        mean_rate = np.mean(sorted_defect_rates)
        ax.axhline(y=mean_rate, color='#e74c3c', linestyle='--', alpha=0.8,
                   linewidth=2, label=f'Mean Rate: {mean_rate:.2f}%')

        # Formatting
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("Batch Size", fontsize=12, fontweight='bold')
        ax.set_ylabel("Defect Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title(f"Batch Size vs. Defect Rate Trend Analysis ({date_str})",
                     fontsize=14, fontweight='bold', pad=20)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        print("✅ Plot created successfully!")
        return fig

    except Exception as e:
        print(f"❌ Error plotting defect rate: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    fig = test_plot_defect_rate()
    if fig:
        print("Saving test plot...")
        fig.savefig("test_plot.png")
        print("✅ Test plot saved as test_plot.png")
    else:
        print("❌ Failed to create test plot") 