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
            SUM(duration_ns) AS total_time_ns,
            AVG(duration_ns) AS avg_time_ns,
            MIN(duration_ns) AS min_time_ns,
            MAX(duration_ns) AS max_time_ns
        FROM per_call_time
        GROUP BY nameId
    ),
    total_time AS (
        SELECT SUM(total_time_ns) AS grand_total_ns FROM agg_stats
    )
    SELECT
        s.value AS Name,
        (a.total_time_ns * 100.0) / t.grand_total_ns AS Percent,
        a.total_time_ns,
        a.num_calls,
        a.avg_time_ns,
        a.min_time_ns,
        a.max_time_ns
    FROM agg_stats a
    JOIN total_time t ON 1=1
    JOIN StringIds s ON s.id = a.nameId
    ORDER BY a.total_time_ns DESC;
    """
    with sqlite3.connect(filepath) as conn:
        df = pd.read_sql_query(query, conn)

    # Remove version suffix from CUDA API calls (e.g., _v3020)
    df["Name"] = df["Name"].str.replace(r"_v\d+$", "", regex=True)

    return df[['Name', 'Percent']]


def plot_kernel_profile(kernel, input_dir, output_dir, filename):
    # Find all relevant sqlite files
    files = [
        f for f in os.listdir(input_dir)
        if f.startswith(f"report_{kernel}_") and f.endswith(".sqlite")
    ]

    # Sort files by extracted <size>
    files = sorted(files, key=lambda f: extract_size_from_filename(f, kernel))

    sizes = []
    op_percent_dict = {}

    for f in files:
        size = extract_size_from_filename(f, kernel)
        if size is None:
            continue
        sizes.append(size)
        df = parse_sqlite_report(os.path.join(input_dir, f))

        for _, row in df.iterrows():
            op = row['Name']
            percent = row['Percent']
            if op not in op_percent_dict:
                op_percent_dict[op] = []
            op_percent_dict[op].append(percent)

    # Pad missing values with 0 to ensure same length
    for op in op_percent_dict:
        while len(op_percent_dict[op]) < len(sizes):
            op_percent_dict[op].append(0.0)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(op_percent_dict, index=sizes)
    plot_df.sort_index(inplace=True)

    # Plot
    ax = plot_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    ax.set_xlabel("Array Size")
    ax.set_ylabel("Time Percentage (%)")
    ax.set_title(f"CUDA Operation Breakdown - {kernel}")
    ax.legend(title="CUDA API Call", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing report_<kernel>_<size>.sqlite files")
    parser.add_argument("output_dir", help="Directory to save plots")

    args = parser.parse_args()

    # Example usage (hardcoded kernel and filename)
    plot_kernel_profile("v0", args.input_dir, args.output_dir, "profile_v0.png")
