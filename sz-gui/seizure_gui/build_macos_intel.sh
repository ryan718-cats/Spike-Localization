#!/usr/bin/env bash
# =============================================================================
# build_macos_intel.sh
#
# Builds a self-contained macOS .app + .dmg targeting x86_64 (Intel) Macs.
#
# HOW ARCHITECTURE TARGETING WORKS
# ─────────────────────────────────
# Apple Silicon Macs ship with Homebrew at /opt/homebrew (ARM only).
# Installing Intel packages requires a SECOND Homebrew at /usr/local.
# This script automates that setup and then builds an x86_64 .app via Rosetta.
#
# ── ONE-TIME SETUP (Apple Silicon host) ──────────────────────────────────────
# Step 1 — Install Rosetta 2 (already done if you ran this before):
#   softwareupdate --install-rosetta
#
# Step 2 — Install the x86_64 Homebrew into /usr/local:
#   arch -x86_64 /bin/bash -c \
#     "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
# Step 3 — Install Python 3.11 (Intel) via that Homebrew:
#   arch -x86_64 /usr/local/bin/brew install python@3.11
#
# Step 4 — Install project dependencies into the Intel Python:
#   arch -x86_64 /usr/local/bin/python3.11 -m pip install \
#     PyQt5 pyqtgraph mne numpy pandas scipy pyinstaller
#
# Then run:   bash seizure_gui/build_macos_intel.sh
# ─────────────────────────────────────────────────────────────────────────────
#
# Requirements (Intel Mac host):
#   • Python 3.10+ with project dependencies + pyinstaller installed normally.
#
# Output:
#   seizure_gui/dist-intel/SeizureAnnotationGUI_intel.app
#   seizure_gui/dist-intel/SeizureAnnotationGUI_intel.dmg
# =============================================================================

set -euo pipefail

APP_NAME="SeizureAnnotationGUI_intel"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist-intel"
BUILD_DIR="$SCRIPT_DIR/build-intel"
ENTRY="$SCRIPT_DIR/sz_gui.py"

echo "──────────────────────────────────────────"
echo "  Seizure Annotation GUI — Intel build"
echo "──────────────────────────────────────────"

# ── Detect host and locate an x86_64 Python ──────────────────────────────────
HOST_ARCH="$(uname -m)"

if [ "$HOST_ARCH" = "arm64" ]; then
    echo "→ Apple Silicon host detected — targeting x86_64 via Rosetta"

    # The x86_64 Homebrew installs Python into /usr/local/bin.
    # Look for python3.11 first, then python3, at that prefix.
    if [ -x "/usr/local/bin/python3.11" ]; then
        PYTHON="arch -x86_64 /usr/local/bin/python3.11"
    elif [ -x "/usr/local/bin/python3" ]; then
        PYTHON="arch -x86_64 /usr/local/bin/python3"
    else
        echo ""
        echo "✗ No x86_64 Python found at /usr/local/bin."
        echo ""
        echo "  Run these steps once to set it up:"
        echo ""
        echo "  1) Install x86_64 Homebrew into /usr/local:"
        echo '     arch -x86_64 /bin/bash -c \'
        echo '       "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        echo ""
        echo "  2) Install Python 3.11 (Intel):"
        echo "     arch -x86_64 /usr/local/bin/brew install python@3.11"
        echo ""
        echo "  3) Install project dependencies:"
        echo "     arch -x86_64 /usr/local/bin/python3.11 -m pip install \\"
        echo "       PyQt5 pyqtgraph mne numpy pandas scipy pyinstaller"
        echo ""
        exit 1
    fi
else
    # Native Intel Mac — plain python3.
    PYTHON="python3"
    echo "→ Intel host detected — building natively"
fi

# ── Verify the chosen Python is actually running as x86_64 ────────────────────
RESOLVED_ARCH=$($PYTHON -c "import platform; print(platform.machine())")
if [ "$RESOLVED_ARCH" != "x86_64" ]; then
    echo ""
    echo "✗ Python reports arch '$RESOLVED_ARCH', expected 'x86_64'."
    echo "  Ensure you followed the one-time setup steps above."
    exit 1
fi
echo "→ Confirmed Python arch: $RESOLVED_ARCH  ($($PYTHON --version 2>&1))"

# ── Ensure PyInstaller is available under the x86_64 Python ──────────────────
if ! $PYTHON -m PyInstaller --version &>/dev/null; then
    echo "→ Installing PyInstaller…"
    $PYTHON -m pip install pyinstaller
fi

# ── Clean previous Intel build artefacts ─────────────────────────────────────
echo "→ Cleaning previous Intel build…"
rm -rf "$BUILD_DIR" "$DIST_DIR"

# ── Build the .app bundle ─────────────────────────────────────────────────────
echo "→ Running PyInstaller (this takes a few minutes)…"
$PYTHON -m PyInstaller \
    --name            "$APP_NAME" \
    --windowed \
    --onedir \
    --clean \
    --noconfirm \
    --distpath        "$DIST_DIR" \
    --workpath        "$BUILD_DIR" \
    --collect-all     mne \
    --collect-all     scipy \
    --collect-all     pyqtgraph \
    --collect-all     mne.io.edf \
    --hidden-import   pandas \
    --hidden-import   pandas._libs.tslibs.np_datetime \
    --hidden-import   pandas._libs.tslibs.nattype \
    --hidden-import   pandas._libs.tslibs.timedeltas \
    --hidden-import   numpy \
    --hidden-import   scipy.signal \
    --hidden-import   scipy.signal._upfirdn \
    --hidden-import   scipy.special._comb \
    --hidden-import   mne.io.edf.edf \
    --hidden-import   mne.filter \
    --osx-bundle-identifier com.seizuregui.annotator.intel \
    --target-arch     x86_64 \
    "$ENTRY"

APP_PATH="$DIST_DIR/$APP_NAME.app"

if [ ! -d "$APP_PATH" ]; then
    echo "✗ Build failed — .app not found at $APP_PATH"
    exit 1
fi

echo "✓ Intel app bundle created: $APP_PATH"

# ── Verify all main binaries are x86_64 ──────────────────────────────────────
MAIN_BIN="$APP_PATH/Contents/MacOS/$APP_NAME"
if [ -f "$MAIN_BIN" ]; then
    BIN_ARCH=$(file "$MAIN_BIN" | grep -o "x86_64\|arm64" | head -1 || true)
    if [ "$BIN_ARCH" = "x86_64" ]; then
        echo "✓ Binary architecture confirmed: x86_64"
    else
        echo "⚠ Warning: binary reports arch '$BIN_ARCH' — verify before distributing"
    fi
fi

# ── Create the .dmg ───────────────────────────────────────────────────────────
DMG_PATH="$DIST_DIR/$APP_NAME.dmg"
echo "→ Creating DMG…"
rm -f "$DMG_PATH"

hdiutil create \
    -volname  "SeizureAnnotationGUI" \
    -srcfolder "$APP_PATH" \
    -ov \
    -format   UDZO \
    "$DMG_PATH"

echo ""
echo "──────────────────────────────────────────"
echo "  Done!"
echo "  App:  $APP_PATH"
echo "  DMG:  $DMG_PATH"
echo ""
echo "  This build targets x86_64 (Intel) Macs."
echo "  It will also run on Apple Silicon via Rosetta 2."
echo "──────────────────────────────────────────"
