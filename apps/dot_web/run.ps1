$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$App = Join-Path $PSScriptRoot "app.py"

Push-Location $RepoRoot
try {
    conda run -n py312 python -m streamlit run $App
}
finally {
    Pop-Location
}

