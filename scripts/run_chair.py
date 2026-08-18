"""Run the original CHAIR scorer without copying third-party code here."""

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--chair-repo", default="./Hallucination")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    scorer = Path(args.chair_repo) / "utils" / "chair.py"
    if not scorer.is_file():
        raise FileNotFoundError(
            f"{scorer} not found; clone https://github.com/LisaAnne/Hallucination.git")
    command = [args.python, str(scorer.resolve()), "--cap_file",
               str(Path(args.captions).resolve()), "--annotation_path",
               str(Path(args.annotation_dir).resolve())]
    subprocess.run(command, check=True, cwd=args.chair_repo)


if __name__ == "__main__":
    main()
