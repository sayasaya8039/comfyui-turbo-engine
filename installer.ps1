# ComfyUI Turbo Engine v2.0.0 — Self-extracting Installer Builder
# Creates a single .exe installer using 7z SFX module

$ErrorActionPreference = "Stop"
$VERSION = "2.0.0"
$PRODUCT = "ComfyUI Turbo Engine"
$EXE_NAME = "comfy-server.exe"
$SRC_DIR = "$PSScriptRoot\release"
$OUT_DIR = "$PSScriptRoot"
$SFX_NAME = "ComfyUI-Turbo-v${VERSION}-setup.exe"

# Create installer config
$configContent = @"
;!@Install@!UTF-8!
Title="$PRODUCT v$VERSION Installer"
BeginPrompt="Install $PRODUCT v$VERSION?"
InstallPath="%ProgramFiles%\\ComfyUI-Turbo"
RunProgram="\"%%T\\$EXE_NAME\""
;!@InstallEnd@!
"@

$configFile = "$OUT_DIR\sfx_config.txt"
Set-Content -Path $configFile -Value $configContent -Encoding UTF8

# Create 7z archive of release files
$archiveFile = "$OUT_DIR\installer_payload.7z"
& 7z a -t7z -mx=9 $archiveFile "$SRC_DIR\*" | Out-Null

# Try to find 7z SFX module
$sfxModule = $null
$sfxPaths = @(
    "$env:ProgramFiles\7-Zip\7z.sfx",
    "$env:SCOOP\apps\7zip\current\7z.sfx",
    "$env:USERPROFILE\scoop\apps\7zip\current\7z.sfx",
    "C:\Users\Owner\scoop\apps\7zip\current\7z.sfx"
)

foreach ($p in $sfxPaths) {
    if (Test-Path $p) {
        $sfxModule = $p
        break
    }
}

if ($sfxModule) {
    Write-Host "Using SFX module: $sfxModule"
    # Combine: SFX module + config + archive = self-extracting installer
    $sfxOutput = "$OUT_DIR\$SFX_NAME"
    $sfxBytes = [System.IO.File]::ReadAllBytes($sfxModule)
    $configBytes = [System.IO.File]::ReadAllBytes($configFile)
    $archiveBytes = [System.IO.File]::ReadAllBytes($archiveFile)

    $output = New-Object byte[] ($sfxBytes.Length + $configBytes.Length + $archiveBytes.Length)
    [System.Buffer]::BlockCopy($sfxBytes, 0, $output, 0, $sfxBytes.Length)
    [System.Buffer]::BlockCopy($configBytes, 0, $output, $sfxBytes.Length, $configBytes.Length)
    [System.Buffer]::BlockCopy($archiveBytes, 0, $output, $sfxBytes.Length + $configBytes.Length, $archiveBytes.Length)
    [System.IO.File]::WriteAllBytes($sfxOutput, $output)

    Write-Host "Created installer: $sfxOutput ($('{0:N1}' -f ((Get-Item $sfxOutput).Length / 1MB)) MB)"
} else {
    Write-Host "7z SFX module not found. ZIP-only release created."
    Write-Host "ZIP: $OUT_DIR\ComfyUI-Turbo-v${VERSION}-win-x64.zip"
}

# Cleanup temp files
Remove-Item -Force $configFile -ErrorAction SilentlyContinue
Remove-Item -Force $archiveFile -ErrorAction SilentlyContinue

Write-Host "Done."
