import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ellipsoid_steering.visualization import synthetic_2d_demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default="synthetic_ellipse.png")
    args = parser.parse_args()
    result = synthetic_2d_demo(args.plot)
    print(result)
    assert result["radius_error"] < 1e-5
    assert result["energy_after"] < result["energy_before"]
