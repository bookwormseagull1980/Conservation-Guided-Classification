"""
QGRAF Installer — Automated download and compilation for Windows
=================================================================

Downloads QGRAF source from the official repository, compiles it
with gfortran (from WinLibs/Mingw-w64), and installs it.

Usage:
    python scripts/install_qgraf.py

Requirements:
    - gfortran (install via winget or manual download)
    - Internet connection for QGRAF source download
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

QGRAF_SOURCE_URL = "http://cfif.ist.utl.pt/~paulo/qgraf/latest/qgraf.tar.gz"
# Mirror (GitHub): https://github.com/apik/QGRAF/archive/refs/heads/master.zip
# If the official site is unreachable, download manually:
#   1. Visit https://github.com/apik/QGRAF
#   2. Download the source (qgraf.f)
#   3. Compile: gfortran -O2 qgraf.f -o qgraf.exe
#   4. Set QGRAF_PATH=<path>/qgraf.exe
QGRAF_INSTALL_DIR = Path.home() / "bin"
QGRAF_BINARY = QGRAF_INSTALL_DIR / "qgraf.exe"


def find_gfortran() -> str | None:
    """Find gfortran on the system."""
    found = shutil.which("gfortran")
    if found:
        return found

    # Common MinGW locations on Windows
    candidates = [
        Path("C:/") / "mingw64" / "bin" / "gfortran.exe",
        Path("C:/") / "mingw32" / "bin" / "gfortran.exe",
        Path("C:/") / "msys64" / "mingw64" / "bin" / "gfortran.exe",
        Path("C:/") / "msys64" / "ucrt64" / "bin" / "gfortran.exe",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    return None


def install_qgraf(install_dir: Path | None = None) -> bool:
    """Download and compile QGRAF.

    Args:
        install_dir: installation directory (default: ~/bin)

    Returns:
        True if installation succeeded
    """
    if install_dir is None:
        install_dir = QGRAF_INSTALL_DIR

    # Check gfortran
    gfortran = find_gfortran()
    if not gfortran:
        print("ERROR: gfortran not found.")
        print("Install MinGW-w64: winget install BrechtSanders.WinLibs.POSIX.UCRT")
        print("Or manually: https://winlibs.com/")
        return False

    print(f"[✓] Found gfortran: {gfortran}")

    # Create install directory
    install_dir.mkdir(parents=True, exist_ok=True)

    # Download QGRAF source
    print(f"[ ] Downloading QGRAF from {QGRAF_SOURCE_URL}...")
    with tempfile.TemporaryDirectory(prefix="qgraf_build_") as build_dir:
        build_path = Path(build_dir)
        tarball = build_path / "qgraf.tar.gz"

        try:
            urllib.request.urlretrieve(QGRAF_SOURCE_URL, str(tarball))
        except Exception as e:
            print(f"ERROR: Download failed: {e}")
            print("Manual download: http://cfif.ist.utl.pt/~paulo/qgraf.html")
            return False

        print(f"[✓] Downloaded to {tarball}")

        # Extract
        import tarfile
        try:
            with tarfile.open(str(tarball), "r:gz") as tf:
                tf.extractall(str(build_path))
            print(f"[✓] Extracted to {build_path}")
        except Exception as e:
            print(f"ERROR: Extraction failed: {e}")
            return False

        # Find the Fortran source
        qgraf_sources = list(build_path.rglob("*.f")) + list(build_path.rglob("*.f90"))
        if not qgraf_sources:
            print("ERROR: No Fortran source files found in archive")
            return False

        qgraf_main = qgraf_sources[0]
        print(f"[ ] Compiling {qgraf_main.name}...")

        # Compile
        output_exe = build_path / "qgraf.exe"
        result = subprocess.run(
            [gfortran, "-O2", "-o", str(output_exe), str(qgraf_main)],
            capture_output=True,
            text=True,
            cwd=str(build_path),
        )

        if result.returncode != 0:
            print(f"ERROR: Compilation failed:")
            print(result.stderr[:1000])
            return False

        print(f"[✓] Compiled: {output_exe}")

        # Install
        target = install_dir / "qgraf.exe"
        shutil.copy2(str(output_exe), str(target))
        print(f"[✓] Installed: {target}")

    # Verify
    result = subprocess.run(
        [str(target), "--version"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        # QGRAF doesn't have --version; try running with no args
        result = subprocess.run(
            [str(target)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    print(f"[✓] QGRAF test run: {result.stderr[:100] if result.stderr else 'OK (no output)'}")

    # Set environment variable hint
    print(f"\nTo add QGRAF to PATH, run:")
    print(f'  setx PATH "%PATH%;{install_dir}"')
    print(f"  Or set QGRAF_PATH={target}")

    return True


def check_qgraf() -> dict:
    """Check QGRAF installation status."""
    from cgc.engine.qgraf_backend import qgraf_status
    return qgraf_status()


if __name__ == "__main__":
    print("=" * 60)
    print("QGRAF Installer for CGC Engine")
    print("=" * 60)
    print()

    status = check_qgraf()
    if status["available"]:
        print(f"[✓] QGRAF already installed: {status['binary_path']}")
        sys.exit(0)

    print(f"[!] QGRAF not found")
    print()

    success = install_qgraf()
    if success:
        print("\n[✓] QGRAF installation complete!")
        print(f"Binary: {QGRAF_BINARY}")
    else:
        print("\n[✗] QGRAF installation failed")
        if status.get("install_hint"):
            print(status["install_hint"])
        sys.exit(1)
