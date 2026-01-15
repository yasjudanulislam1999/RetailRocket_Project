from __future__ import annotations

import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
load_dotenv()



def upload_file(s3_client, file_path: Path, bucket: str, key: str) -> None:
    """
    Upload one file to S3.

    bucket = S3 bucket name
    key    = path inside the bucket (like folders + filename)
    """
    s3_client.upload_file(str(file_path), bucket, key)


def main() -> None:
    # Read settings from environment
    region = os.getenv("AWS_REGION", "eu-west-2")
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX", "retailrocket-item2item")

    if not bucket:
        print("ERROR: S3_BUCKET is not set in your environment/.env")
        print("Fix: add S3_BUCKET=your-bucket-name to .env")
        return

    # Files to upload
    topk_path = Path("artefacts/topk.json")
    report_path = Path("artefacts/eval_report.json")

    for p in [topk_path, report_path]:
        if not p.exists():
            print(f"ERROR: missing file: {p}")
            print("Fix: build artefacts first (topk + evaluate).")
            return

    # Create an S3 client (uses your AWS credentials)
    s3 = boto3.client("s3", region_name=region)

    # Build S3 keys (paths inside bucket)
    topk_key = f"{prefix}/topk.json"
    report_key = f"{prefix}/eval_report.json"

    print("Uploading to S3...")
    print(f" - {topk_path}  ->  s3://{bucket}/{topk_key}")
    upload_file(s3, topk_path, bucket, topk_key)

    print(f" - {report_path}  ->  s3://{bucket}/{report_key}")
    upload_file(s3, report_path, bucket, report_key)

    print("\nDONE ✅")
    print("Uploaded artefacts:")
    print(f"s3://{bucket}/{topk_key}")
    print(f"s3://{bucket}/{report_key}")


if __name__ == "__main__":
    main()
