import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle


# Sort versions by assumed version order
def version_sort_key(v):
    if v == "serial":
        return -1
    if v.startswith("v") and v[1:].isdigit():
        return int(v[1:])
    return float('inf')


def plot_execution_time_vs_size(input_path, output_path, partition):
    # Load CSV data
    df = pd.read_csv(input_path)

    # Validate expected columns
    if not {'q', 'version', 'time_ms'}.issubset(df.columns):
        raise ValueError("CSV must have columns: q, version, time_ms")

    # Check for valid time values
    if (df["time_ms"] <= 0).any():
        raise ValueError("time_ms must be strictly positive for log-scale plotting.")

    # Sort and group
    df.sort_values(by=["version", "q"], inplace=True)

    versions = sorted(df["version"].unique(), key=version_sort_key)

    # Marker styles (repeats if more than 10 versions)
    marker_styles = cycle(['o', 's', '^', 'v', 'D', '*', 'P', 'X', '<', '>'])

    # Plot
    plt.figure(figsize=(10, 6))
    for version in versions:
        sub_df = df[df["version"] == version]
        if not sub_df.empty:
            marker = next(marker_styles)
            plt.plot(
                sub_df["q"],
                sub_df["time_ms"],
                marker=marker,
                label=version.upper()
            )

    plt.xlabel("Array Size (N = 2^q)")
    plt.ylabel("Total Execution Time (ms)")
    plt.title(f"Total Execution Time vs Array Size (Partition: {partition.upper()})")
    plt.legend(loc='upper left')
    plt.grid(True)

    q_values = sorted(df["q"].unique())
    plt.xticks(ticks=q_values, labels=[f"$2^{{{q}}}$" for q in q_values])
    plt.yscale("log", base=10)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


def export_speedup_table(input_path, output_path, q_value):
    # Load CSV data
    df = pd.read_csv(input_path)

    # Validate expected columns
    if not {'q', 'version', 'time_ms'}.issubset(df.columns):
        raise ValueError("CSV must have columns: q, version, time_ms")

    # Filter by q
    q_df = df[df["q"] == q_value].copy()
    if q_df.empty:
        raise ValueError(f"No data found for q = {q_value}")

    q_df.sort_values(by="version", key=lambda col: col.map(version_sort_key), inplace=True)

    # Compute speedups
    times = q_df["time_ms"].values
    versions = q_df["version"].tolist()

    step_speedups = [""]
    cumulative_speedups = [1.0]

    for i in range(1, len(times)):
        step = times[i - 1] / times[i]
        cum = cumulative_speedups[i - 1] * step
        step_speedups.append(f"{step:.2f}")
        cumulative_speedups.append(cum)

    # Prepare output table
    result_df = pd.DataFrame({
        "Version": [v.upper() for v in versions],
        "Time (ms)": [f"{t:.3f}" for t in times],
        "Step Speedup": step_speedups,
        "Cumulative Speedup": [f"{s:.2f}" if isinstance(s, float) else s for s in cumulative_speedups]
    })

    # Export
    result_df.to_csv(output_path, index=False)
    print(f"Saved speedup table to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Full path to input CSV file")
    parser.add_argument("export_dir", help="Directory to export plots and tables")
    parser.add_argument("partition", help="SLURM job partition name (e.g., gpu, ampere)")
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    plot_filename = os.path.join(args.export_dir, f"execution_times_{args.partition}.png")
    plot_execution_time_vs_size(args.input_path, plot_filename, args.partition)

    q_value = 27
    table_filename = os.path.join(args.export_dir, f"speedup_table_{args.partition}_{q_value}.dat")
    export_speedup_table(args.input_path, table_filename, q_value)
