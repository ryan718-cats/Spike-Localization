# Launch spike review GUI using environment variables (no passwords in this file).
# In PowerShell, set once per session:
#   $env:IEEG_USERNAME = "your_ieeg_username"
#   $env:IEEG_PASSWORD = "your_ieeg_password"
Set-Location $PSScriptRoot

if (-not $env:IEEG_USERNAME) {
    Write-Error "Set `$env:IEEG_USERNAME before running."
    exit 1
}
if (-not $env:IEEG_PASSWORD) {
    Write-Error "Set `$env:IEEG_PASSWORD before running (not stored in this script)."
    exit 1
}

python run_spike_review_gui.py --ieeg-username $env:IEEG_USERNAME
