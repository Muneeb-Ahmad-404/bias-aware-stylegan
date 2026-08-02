import subprocess
from pathlib import Path


def main():
    script = Path(__file__).parent / "setup_colab.sh"

    subprocess.run(["bash", str(script)], check=True)


if __name__ == "__main__":
    main()
