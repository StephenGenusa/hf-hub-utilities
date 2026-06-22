#!/usr/bin/env python3
"""
hfu.py - Hugging Face URL downloader
Transforms Hugging Face URLs into 'hf download' commands and executes them.

Usage:
    hfu.py <url_or_path>

Supported formats:
    Full URL:   https://huggingface.co/<namespace>/<model>
                https://huggingface.co/datasets/<namespace>/<name>
                https://huggingface.co/spaces/<namespace>/<name>
    Partial:    <namespace>/<model>
                datasets/<namespace>/<name>
                models/<namespace>/<name>
                spaces/<namespace>/<name>

Examples:
    hfu.py https://huggingface.co/zai-org/GLM-OCR
    hfu.py datasets/davanstrien/enc-brit-glm-ocr-v2-full
    hfu.py zai-org/GLM-OCR
"""

import sys
import subprocess
import re
import os


def parse_hf_input(input_str: str) -> tuple[str, str]:
    """
    Parse Hugging Face URL or path and return (repo_id, repo_type).

    Args:
        input_str: URL or path like:
            - https://huggingface.co/datasets/davanstrien/enc-brit-glm-ocr-v2-full
            - datasets/davanstrien/enc-brit-glm-ocr-v2-full
            - davanstrien/enc-brit-glm-ocr-v2-full

    Returns:
        Tuple of (repo_id, repo_type) where repo_type is 'model', 'dataset', or 'space'
    """
    input_str = input_str.strip()

    # Detect repo type from prefix (datasets/, models/, spaces/)
    repo_type = "model"  # default
    path_part = input_str

    if input_str.startswith("datasets/"):
        repo_type = "dataset"
        path_part = input_str[len("datasets/"):]
    elif input_str.startswith("models/"):
        repo_type = "model"
        path_part = input_str[len("models/"):]
    elif input_str.startswith("spaces/"):
        repo_type = "space"
        path_part = input_str[len("spaces/"):]
    elif input_str.startswith("https://huggingface.co/"):
        # Full URL - extract path after domain
        path_part = input_str[len("https://huggingface.co/"):]
    elif input_str.startswith("http://huggingface.co/"):
        # Full URL - extract path after domain
        path_part = input_str[len("http://huggingface.co/"):]

        # Check for explicit repo type in URL
        if path_part.startswith("datasets/"):
            repo_type = "dataset"
            path_part = path_part[len("datasets/"):]
        elif path_part.startswith("models/"):
            repo_type = "model"
            path_part = path_part[len("models/"):]
        elif path_part.startswith("spaces/"):
            repo_type = "space"
            path_part = path_part[len("spaces/"):]
        # else: assume model

    # Validate format: should be namespace/name (allow dots, dashes, underscores)
    # e.g., davanstrien/enc-brit-glm-ocr-v2-full
    pattern = r"^([\w\-\.]+)/([\w\-\.]+)$"
    match = re.match(pattern, path_part)

    if not match:
        raise ValueError(
            f"Invalid Hugging Face path format: '{input_str}'\n"
            f"Expected: <namespace>/<name> (e.g., davanstrien/enc-brit-glm-ocr-v2-full)"
        )

    repo_id = path_part
    return repo_id, repo_type


def main():
    if len(sys.argv) != 2:
        print("❌ Error: Exactly one URL argument required", file=sys.stderr)
        print(f"Usage: {sys.argv[0]} <url_or_path>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  hfu.py https://huggingface.co/zai-org/GLM-OCR", file=sys.stderr)
        print("  hfu.py datasets/davanstrien/enc-brit-glm-ocr-v2-full", file=sys.stderr)
        print("  hfu.py zai-org/GLM-OCR", file=sys.stderr)
        sys.exit(1)

    input_str = sys.argv[1]

    try:
        repo_id, repo_type = parse_hf_input(input_str)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    # Build command
    cmd = ["hf", "download", repo_id]

    # Add repo type if not model (model is the default)
    if repo_type != "model":
        cmd.extend(["--repo-type", repo_type])

    # Show what we're doing
    print(f"📦 Downloading {repo_type}: {repo_id}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()

    try:
        # Execute command with live output streaming (critical for download progress)
        result = subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"\n✅ Successfully downloaded")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    except FileNotFoundError:
        print("❌ 'hf' CLI not found. Install with:", file=sys.stderr)
        print("   pip install 'huggingface_hub[cli]'  OR", file=sys.stderr)
        print("   curl -LsSf https://hf.co/cli/install.sh | bash", file=sys.stderr)
        sys.exit(127)


if __name__ == "__main__":
    main()