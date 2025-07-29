import os
import re
import sqlite3
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def extract_size_from_filename(filename, kernel):
    pattern = rf"report_{re.escape(kernel)}_(\d+)\.sqlite"
    match = re.match(pattern, filename)
    return int(match.group(1)) if match else None


def parse_sqlite_report(filepath):
    query = """
    WITH per_call_time AS (
        SELECT
            nameId,
            (end - start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
    ),
    agg_stats AS (
        SELECT
            nameId,
            COUNT(*) AS num_calls,
            SUM(duration_ns) AS total_time_ns
        FROM per_call_time
        GROUP BY nameId
    )
    SELECT
        s.value AS Name,
        a.total_time_ns / 1e6 AS TimeMs  -- Convert to ms
    FROM agg_stats a
    JOIN StringIds s ON s.id = a.nameId
    ORDER BY a.total_time_ns DESC;
    """
    with sqlite3.connect(filepath) as conn:
        df = pd.read_sql_query(query, conn)

    # Clean CUDA API names
    df["Name"] = df["Name"].str.replace(r"_v\d+$", "", regex=True)

    return df[['Name', 'TimeMs']]


def plot_kernel_time_breakdown(kernel, input_dir, output_dir, filename):
    files = [
        f for f in os.listdir(input_dir)
        if f.startswith(f"report_{kernel}_") and f.endswith(".sqlite")
    ]

    files = sorted(files, key=lambda f: extract_size_from_filename(f, kernel))

    sizes = []
    op_time_dict = {}

    for f in files:
        size = extract_size_from_filename(f, kernel)
        if size is None:
            continue
        sizes.append(size)
        df = parse_sqlite_report(os.path.join(input_dir, f))

        for _, row in df.iterrows():
            op = row['Name']
            time_ms = row['TimeMs']
            if op not in op_time_dict:
                op_time_dict[op] = []
            op_time_dict[op].append(time_ms)

    # Pad missing values with 0s
    for op in op_time_dict:
        while len(op_time_dict[op]) < len(sizes):
            op_time_dict[op].append(0.0)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(op_time_dict, index=sizes)
    plot_df.sort_index(inplace=True)

    # Plot
    ax = plot_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    ax.set_xlabel("Array Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"CUDA Operation Breakdown (Absolute Time) - {kernel}")
    ax.legend(title="CUDA API Call", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_total_execution_time_vs_size(csv_path, output_dir, output_filename):

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Validate expected columns
    if not {'q', 'kernel', 'time'}.issubset(df.columns):
        raise ValueError("CSV must have columns: q, kernel, time")

    # Compute array size and convert time to nanoseconds
    df["Size"] = 2 ** df["q"]
    df["Time (ns)"] = df["time"] * 1e9

    # Sort and group
    df.sort_values(by=["kernel", "Size"], inplace=True)

    # Plot
    plt.figure(figsize=(10, 6))
    for kernel in ["none", "v0", "v1", "v2"]:
        sub_df = df[df["kernel"] == kernel]
        if not sub_df.empty:
            plt.plot(
                sub_df["Size"],
                sub_df["Time (ns)"],
                marker='o',
                label=kernel.upper()
            )

    plt.xlabel("Array Size (N = 2^q)")
    plt.ylabel("Total Execution Time (ns)")
    plt.title("Total Execution Time vs Array Size")
    plt.legend()
    plt.grid(True)
    plt.xscale("log", base=2)
    plt.yscale("log")  # Optional: useful if large scale differences
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing report_<kernel>_<size>.sqlite files")
    parser.add_argument("output_dir", help="Directory to save plots")

    args = parser.parse_args()

    # Example usage (hardcoded kernel and filename)
    plot_kernel_time_breakdown("v0", args.input_dir, args.output_dir, "time_breakdown_v0.png")
    plot_kernel_time_breakdown("v1", args.input_dir, args.output_dir, "time_breakdown_v1.png")
    plot_total_execution_time_vs_size(os.path.join(args.input_dir, "total_times.log"), args.output_dir, "total_execution_time_vs_size.png")
