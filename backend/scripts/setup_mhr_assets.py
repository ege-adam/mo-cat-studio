#!/usr/bin/env python3
"""
Post-install script to download and set up MHR assets.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil

MHR_ASSETS_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"
ASSETS_ZIP = "assets.zip"


def get_site_packages_dir():
    """Get the site-packages directory for the current Python environment."""
    import site
    site_packages = site.getsitepackages()
    if site_packages:
        return Path(site_packages[0])
    # Fallback for virtual environments
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    raise RuntimeError("Could not determine site-packages directory")


def download_assets(output_path: Path):
    """Download MHR assets zip file."""
    print(f"[MHR Setup] Downloading assets from {MHR_ASSETS_URL}...")
    try:
        urllib.request.urlretrieve(MHR_ASSETS_URL, output_path)
        print(f"[MHR Setup] Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"[MHR Setup] Failed to download assets: {e}")
        return False


def extract_assets(zip_path: Path, extract_to: Path):
    """Extract assets zip to target directory."""
    print(f"[MHR Setup] Extracting assets to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[MHR Setup] Extracted successfully")
        return True
    except Exception as e:
        print(f"[MHR Setup] Failed to extract assets: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("MHR Assets Setup")
    print("=" * 60)
    
    # Get site-packages directory
    try:
        site_packages = get_site_packages_dir()
        print(f"[MHR Setup] Site-packages: {site_packages}")
    except RuntimeError as e:
        print(f"[MHR Setup] Error: {e}")
        return 1
    
    # Target assets directory
    assets_dir = site_packages / "assets"
    
    # Check if assets already exist
    lod1_fbx = assets_dir / "lod1.fbx"
    if lod1_fbx.exists():
        print(f"[MHR Setup] Assets already exist at {assets_dir}")
        print("[MHR Setup] Skipping download")
        return 0
    
    # Download assets
    temp_dir = Path("/tmp/mhr_setup")
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / ASSETS_ZIP
    
    if not download_assets(zip_path):
        return 1
    
    # Extract assets
    if not extract_assets(zip_path, site_packages):
        return 1
    
    # Cleanup
    print(f"[MHR Setup] Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Verify installation
    if lod1_fbx.exists():
        print(f"[MHR Setup] ✓ Assets successfully installed at {assets_dir}")
        return 0
    else:
        print(f"[MHR Setup] ✗ Assets installation failed - lod1.fbx not found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
