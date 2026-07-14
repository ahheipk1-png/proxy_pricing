import os

from ExoticPipelineCore import print_run_summary, run_proxy_study


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "SingleExoticOptExperiment", "results")
MARKDOWN_DIR = os.path.join(ROOT, "Markdown", "SingleExotic", "results")


def main():
    rows, elapsed = run_proxy_study(
        asset_count=1,
        output_dir=OUTPUT_DIR,
        markdown_dir=MARKDOWN_DIR,
        label="Single",
    )
    print_run_summary("Single-underlying", rows, elapsed)
    print()
    print(f"summary written to: {os.path.join(MARKDOWN_DIR, 'summary.md')}")
    print(f"results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
