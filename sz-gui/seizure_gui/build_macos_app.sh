#!/usr/bin/env bash
# =============================================================================
# build_macos_app.sh
#
# Builds a self-contained macOS .app bundle for the Seizure Annotation GUI
# using PyInstaller, then wraps it in a distributable .dmg using hdiutil
# (built into macOS — no third-party tools required).
#
# Requirements:
#   • Python 3.10+ with the project dependencies installed
#     (pip install -r seizure_gui/requirements.txt)
#   • PyInstaller (installed automatically below if missing)
#
# Usage (run from the project root):
#   bash seizure_gui/build_macos_app.sh
#
# Output:
#   seizure_gui/dist/SeizureAnnotationGUI.app   ← the application bundle
#   seizure_gui/dist/SeizureAnnotationGUI.dmg   ← distributable disk image
#
# Notes:
#   • The .dmg is built for the architecture of the machine running this
#     script (Apple Silicon or Intel).  To target the other architecture,
#     run this script under Rosetta or on the target hardware.
#   • Annotations are saved to ~/Documents/SeizureAnnotations/ at runtime.
#   • The SOZ electrode list is baked into the app; no CSV file is needed.
# =============================================================================

set -euo pipefail

APP_NAME="SeizureAnnotationGUI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"
BUILD_DIR="$SCRIPT_DIR/build"
ENTRY="$SCRIPT_DIR/sz_gui.py"

echo "──────────────────────────────────────────"
echo "  Seizure Annotation GUI — macOS build"
echo "──────────────────────────────────────────"

# ── 1. Ensure PyInstaller is available ────────────────────────────────────────
if ! python3 -m PyInstaller --version &>/dev/null; then
    echo "→ Installing PyInstaller…"
    pip install pyinstaller
fi

# ── 2. Clean previous build artefacts ────────────────────────────────────────
echo "→ Cleaning previous build…"
rm -rf "$BUILD_DIR" "$DIST_DIR"

# ── 3. Build the .app bundle ──────────────────────────────────────────────────
echo "→ Running PyInstaller (this takes a few minutes)…"
python3 -m PyInstaller \
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
    --osx-bundle-identifier com.seizuregui.annotator \
    "$ENTRY"

APP_PATH="$DIST_DIR/$APP_NAME.app"

if [ ! -d "$APP_PATH" ]; then
    echo "✗ Build failed — .app not found at $APP_PATH"
    exit 1
fi

echo "✓ App bundle created: $APP_PATH"

# ── 4. Create the .dmg ────────────────────────────────────────────────────────
DMG_PATH="$DIST_DIR/$APP_NAME.dmg"
echo "→ Creating DMG…"

# Remove stale DMG if present
rm -f "$DMG_PATH"

hdiutil create \
    -volname  "$APP_NAME" \
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
echo "  Distribute the .dmg.  Users double-click it,"
echo "  drag the app to /Applications, and launch —"
echo "  no Python installation required."
echo "──────────────────────────────────────────"
