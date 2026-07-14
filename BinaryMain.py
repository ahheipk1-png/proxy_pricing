import os

from ExoticPipelineCore import print_family_run_summary, run_family_proxy_study


ROOT = os.path.dirname(os.path.abspath(__file__))
FAMILY = "Binary"
OUTPUT_DIR = os.path.join(ROOT, "BinaryOptExperiment", "results")
MARKDOWN_DIR = os.path.join(ROOT, "Markdown", "Binary", "results")


def main():
    rows, elapsed = run_family_proxy_study(FAMILY, OUTPUT_DIR, MARKDOWN_DIR, path_count=131_072)
    print_family_run_summary(FAMILY, rows, elapsed)
    print()
    print(f"summary written to: {os.path.join(MARKDOWN_DIR, 'summary.md')}")
    print(f"results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
