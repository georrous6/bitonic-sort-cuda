import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle


def plot_execution_time_vs_size(csv_path, output_dir, output_filename):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Validate expected columns
    if not {'q', 'kernel', 'time_ms'}.issubset(df.columns):
        raise ValueError("CSV must have columns: q, kernel, time_ms")

    # Check for valid time values
    if (df["time_ms"] <= 0).any():
        raise ValueError("time_ms must be strictly positive for log-scale plotting.")

    # Sort and group
    df.sort_values(by=["kernel", "q"], inplace=True)

    # Determine unique kernel versions and sort them meaningfully
    def kernel_sort_key(k):
        if k == "none":
            return -1
        if k.startswith("v") and k[1:].isdigit():
            return int(k[1:])
        return float('inf')

    kernel_versions = sorted(df["kernel"].unique(), key=kernel_sort_key)

    # Marker styles (repeats if more than 10 kernels)
    marker_styles = cycle(['o', 's', '^', 'v', 'D', '*', 'P', 'X', '<', '>'])

    # Plot
    plt.figure(figsize=(10, 6))
    for kernel in kernel_versions:
        sub_df = df[df["kernel"] == kernel]
        if not sub_df.empty:
            marker = next(marker_styles)
            plt.plot(
                sub_df["q"],
                sub_df["time_ms"],
                marker=marker,
                label=kernel.upper()
            )

    plt.xlabel("Array Size (N = 2^q)")
    plt.ylabel("Total Execution Time (ms)")
    plt.title("Total Execution Time vs Array Size")
    plt.legend()
    plt.grid(True)

    q_values = sorted(df["q"].unique())
    plt.xticks(ticks=q_values, labels=[f"$2^{{{q}}}$" for q in q_values])
    plt.yscale("log", base=10)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


def export_speedup_table(csv_path, data_dir, q_value, output_filename):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Validate expected columns
    if not {'q', 'kernel', 'time_ms'}.issubset(df.columns):
        raise ValueError("CSV must have columns: q, kernel, time_ms")

    # Filter by q
    q_df = df[df["q"] == q_value].copy()
    if q_df.empty:
        raise ValueError(f"No data found for q = {q_value}")

    # Sort kernels by assumed version order
    def kernel_sort_key(k):
        if k == "none":
            return -1
        if k.startswith("v") and k[1:].isdigit():
            return int(k[1:])
        return float('inf')

    q_df.sort_values(by="kernel", key=lambda col: col.map(kernel_sort_key), inplace=True)

    # Compute speedups
    times = q_df["time_ms"].values
    kernels = q_df["kernel"].tolist()

    step_speedups = [""]
    cumulative_speedups = [1.0]

    for i in range(1, len(times)):
        step = times[i - 1] / times[i]
        cum = cumulative_speedups[i - 1] * step
        step_speedups.append(f"{step:.2f}")
        cumulative_speedups.append(cum)

    # Prepare output table
    result_df = pd.DataFrame({
        "Kernel Version": [k.upper() for k in kernels],
        "Time (ms)": [f"{t:.3f}" for t in times],
        "Step Speedup": step_speedups,
        "Cumulative Speedup": [f"{s:.2f}" if isinstance(s, float) else s for s in cumulative_speedups]
    })

    # Export
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, output_filename)
    result_df.to_csv(output_path, index=False)
    print(f"Saved speedup table to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Full path to input CSV file")
    parser.add_argument("plot_dir", help="Directory to save plot (.png)")
    parser.add_argument("data_dir", help="Directory to save data files (.dat)")
    args = parser.parse_args()

    plot_execution_time_vs_size(args.csv_path, args.plot_dir, "execution_time_vs_size.png")
    export_speedup_table(args.csv_path, args.data_dir, 27, "speedup_table.dat")
