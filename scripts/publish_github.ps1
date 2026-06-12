param(
  [string]$RepoName = "hd-tilag-reproduction",
  [ValidateSet("public", "private")]
  [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI (`gh`) is required. Install it or use the manual git commands in README.md."
}

if (-not (Test-Path ".git")) {
  git init
}

$branch = (git branch --show-current)
if (-not $branch) {
  git checkout -b main
}

git add .
git commit -m "Reproduce HD-TILAG"
gh repo create $RepoName "--$Visibility" --source . --remote origin --push
