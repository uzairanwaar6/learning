# Define the version and installer URL
$pythonVersion = "3.12.3"
$installerUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
$installerPath = "\python-installer.exe"

Write-Host "Downloading Python $pythonVersion..."
# Download the installer
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

Write-Host "Download complete. Installing Python..."
# Run the installer silently
Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait

Write-Host "Installation complete. Verifying..."
# Verify installation
python --version
