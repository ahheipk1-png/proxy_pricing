from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Documentation" / "proxy_pricing_methodology.tex"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "proxy_pricing_methodology.pdf"
LOCAL_TECTONIC = ROOT / "tools" / "tectonic" / "tectonic.exe"


def find_tectonic():
    if LOCAL_TECTONIC.exists():
        return str(LOCAL_TECTONIC)
    path_exe = shutil.which("tectonic")
    if path_exe:
        return path_exe
    raise SystemExit(
        "Tectonic is required to build the textbook PDF.\n"
        "Install it from https://github.com/tectonic-typesetting/tectonic/releases "
        "or place tectonic.exe at tools/tectonic/tectonic.exe."
    )


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tectonic = find_tectonic()
    subprocess.run(
        [tectonic, "--outdir", str(OUTPUT_DIR), str(SOURCE)],
        cwd=str(ROOT),
        check=True,
    )
    print(OUTPUT)


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
