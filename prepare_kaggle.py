import argparse
import json
from pathlib import Path

from src.config import load_config
from src.validate_output import validate_output


def main():
    parser = argparse.ArgumentParser(description="Validate output and create Kaggle dataset metadata")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--owner", required=True, help="Kaggle username or organization")
    parser.add_argument("--slug", required=True, help="URL-safe dataset slug")
    parser.add_argument("--title", required=True)
    parser.add_argument("--license", default="other", dest="license_name")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = validate_output(
        config.data.manifest_path, config.data.sequence_dir, require_all_splits=True
    )
    output_dir = Path(config.data.output_dir)
    metadata = {
        "title": args.title,
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": args.license_name}],
        "isPrivate": not args.public,
        "description": (
            "Preprocessed face sequences. Use manifests/sequence_manifest.csv "
            "and preserve its leakage-safe split column."
        ),
    }
    path = output_dir / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": str(path), "validation": result}, indent=2))
    public_flag = " --public" if args.public else ""
    print(
        f"Upload with: kaggle datasets create -p \"{output_dir}\" "
        f"-r zip{public_flag}"
    )


if __name__ == "__main__":
    main()
