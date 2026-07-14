import os

from ExoticPipelineCore import print_run_summary, run_proxy_study


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "BasketExoticOptExperiment", "results")
MARKDOWN_DIR = os.path.join(ROOT, "Markdown", "BasketExotic", "results")


def main():
    rows, elapsed = run_proxy_study(
        asset_count=4,
        output_dir=OUTPUT_DIR,
        markdown_dir=MARKDOWN_DIR,
        label="Basket",
        path_count=65_536,
        train_state_count=241,
        validation_state_count=101,
        product_limit=120,
        scale_batch=8,
    )
    print_run_summary("Basket", rows, elapsed)
    print()
    print(f"summary written to: {os.path.join(MARKDOWN_DIR, 'summary.md')}")
    print(f"results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
