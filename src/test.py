from pathlib import Path


def describe(path: Path) -> None:
    print(f"\nChecking: {path}")
    print(f"Resolved: {path.resolve()}")
    print(f"Exists: {path.exists()}")
    print(f"Is dir: {path.is_dir()}")
    if path.exists() and path.is_dir():
        try:
            items = list(path.iterdir())[:10]
            print(f"First {len(items)} item(s):")
            for item in items:
                kind = "DIR " if item.is_dir() else "FILE"
                print(f"  [{kind}] {item.name}")
        except Exception as exc:
            print(f"Could not list contents: {exc}")


def main() -> None:
    here = Path(__file__).resolve().parent

    candidate_paths = [
        here / "../data",
        here / "../data/raw_data",
        here / "../data/converted_data",
        here / "../data/simulated",
    ]

    print("Testing access to Azure Quantum Parity Readout dataset paths...")
    print(f"Script location: {here}")

    for raw_path in candidate_paths:
        describe(raw_path)

    print("\nDone.")


if __name__ == "__main__":
    main()