from pathlib import Path
import subprocess


def main():
    script = Path(__file__).parent / "setup_colab.sh"

    subprocess.run(
        ["bash", str(script)],
        check=True
    )


if __name__ == "__main__":
    main()