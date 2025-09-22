<#
fill_cred_secret.ps1

This script helps judges quickly get their credentials set up so the notebook can actually run.

What it does:
- Creates the src/cred_secret.txt file that the notebook needs, using one of these methods:
  - Point it to an existing service account key file and it'll copy the JSON
  - Paste your JSON key content directly into the terminal
  - Let it create a new service account key through gcloud and grab the JSON automatically
- Optionally handles all the boring IAM setup so the notebook works end-to-end:
  - Makes sure you're logged into gcloud with an active project
  - Turns on the services you need (BigQuery, BigQuery Connection, Vertex AI)
  - Creates the demo datasets if they don't exist yet (mskg_demo, mskg_staging by default)
  - Gives your service account the right permissions on those datasets
  - Sets up the 'gemini' BigQuery connection and makes sure your service account can use it
  - Handles the Vertex AI permissions for the BigQuery service agent

How to run it (from the repo root directory):
  powershell -ExecutionPolicy Bypass -File scripts\fill_cred_secret.ps1
#>

[CmdletBinding()] param()

function Write-Separator {
  Write-Host ('-' * 60) 
}

function Confirm-Or-Prompt($Message, $DefaultYes=$true) {
  $suffix = if ($DefaultYes) { '[Y/n]' } else { '[y/N]' }
  $resp = Read-Host "$Message $suffix"
  if (-not $resp) { return $DefaultYes }
  return ($resp -match '^(y|Y)')
}

function Try-Gcloud-BqConn {
  param([string[]]$Args)
  $channels = @('', 'beta', 'alpha')
  foreach ($ch in $channels) {
    try {
      if ($ch -eq '') {
        & gcloud bigquery connections @Args 2>$null | Out-Null
      } else {
        & gcloud $ch bigquery connections @Args 2>$null | Out-Null
      }
      if ($LASTEXITCODE -eq 0) { return $true }
    } catch {}
  }
  return $false
}

function Get-Gcloud-BqConnOutput {
  param([string[]]$Args)
  $channels = @('', 'beta', 'alpha')
  foreach ($ch in $channels) {
    try {
      $out = $null
      if ($ch -eq '') {
        $out = & gcloud bigquery connections @Args 2>$null
      } else {
        $out = & gcloud $ch bigquery connections @Args 2>$null
      }
      if ($LASTEXITCODE -eq 0 -and $out -ne $null) { return $out }
    } catch {}
  }
  return $null
}

function Add-ConnIamMemberRest {
  param(
    [string]$ProjectId,
    [string]$Region,
    [string]$ConnectionId,
    [string]$Member,
    [string]$Role
  )
  try {
    $token = (& gcloud auth print-access-token 2>$null)
    if (-not $token) { return $false }
    $base = "https://bigqueryconnection.googleapis.com/v1/projects/$ProjectId/locations/$Region/connections/$ConnectionId"
    $headers = @{ 'Authorization' = "Bearer $token"; 'Content-Type' = 'application/json' }
    $iam = Invoke-RestMethod -Method GET -Headers $headers -Uri ($base + ':getIamPolicy') -ErrorAction SilentlyContinue
    if (-not $iam) { $iam = @{ bindings = @(); etag = '' } }
    if (-not $iam.bindings) { $iam.bindings = @() }
    $updated = $false
    foreach ($b in $iam.bindings) {
      if ($b.role -eq $Role) {
        if (-not $b.members) { $b.members = @() }
        if (-not ($b.members -contains $Member)) { $b.members += $Member; $updated = $true }
      }
    }
    if (-not ($iam.bindings | Where-Object { $_.role -eq $Role })) {
      $iam.bindings += @{ role = $Role; members = @($Member) }
      $updated = $true
    }
    if (-not $updated) { return $true }
    $policy = @{ bindings = $iam.bindings }
    if ($iam.etag) { $policy.etag = $iam.etag }
    $body = @{ policy = $policy } | ConvertTo-Json -Depth 10
    $resp = Invoke-RestMethod -Method POST -Headers $headers -Uri ($base + ':setIamPolicy') -Body $body -ErrorAction SilentlyContinue
    return $true
  } catch { return $false }
}

try {
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
  $credFile = Join-Path $repoRoot 'src\cred_secret.txt'
  $srcDir = Join-Path $repoRoot 'src'
  if (-not (Test-Path $srcDir)) { New-Item -ItemType Directory -Path $srcDir | Out-Null }
}
catch {
  Write-Error "Failed to locate repo root: $_"
  exit 1
}

Write-Separator
Write-Host "Project Anchor - Credential & IAM Helper" -ForegroundColor Cyan
Write-Separator
Write-Host "Pick an option:" -ForegroundColor Yellow
Write-Host "  [1] Use an existing service account key (enter path)"
Write-Host "  [2] Paste JSON key content now (multi-line)"
Write-Host "  [3] Create a key via gcloud (auto-setup)"

$choice = Read-Host "Enter 1 / 2 / 3"

# These might get set during option 3 and reused by IAM bootstrap
$createdProjectId = ''
$createdSaEmail = ''

switch ($choice) {
  '1' {
    $path = Read-Host "Enter full path to your service account JSON (e.g., D:\\secrets\\my-sa.json)"
    if (-not (Test-Path $path)) {
      Write-Error "File not found: $path"
      exit 1
    }
    try {
      # validate and copy JSON content into cred_secret.txt
      $json = Get-Content -Raw -Path $path
      $null = $json | ConvertFrom-Json
    } catch {
      Write-Error "The file doesn't look like valid JSON: $_"
      exit 1
    }
    
    # Make sure the JSON content gets saved properly
    try {
      Set-Content -Path $credFile -Value $json -Encoding UTF8
      Write-Host "Wrote JSON content to $credFile" -ForegroundColor Green
      
      # Double-check the file was written correctly
      if (-not (Test-Path $credFile) -or (Get-Item $credFile).Length -eq 0) {
        throw "Failed to write content to $credFile"
      }
      
      $secretsDir = Join-Path $repoRoot '.secrets'
      if (-not (Test-Path $secretsDir)) { New-Item -ItemType Directory -Path $secretsDir | Out-Null }
      $saJsonPath = Join-Path $secretsDir 'gcp_sa.json'
      Set-Content -Path $saJsonPath -Value $json -Encoding UTF8
      Write-Host "Wrote JSON content to $saJsonPath" -ForegroundColor Green
      
      # Make sure this one worked too
      if (-not (Test-Path $saJsonPath) -or (Get-Item $saJsonPath).Length -eq 0) {
        throw "Failed to write content to $saJsonPath"
      }
    } catch {
      Write-Error "Failed to save JSON content: $_"
      exit 1
    }
  }
  '2' {
    Write-Host "Paste your JSON key content below. Press Ctrl+Z then Enter to finish (Ctrl+D in some terminals)." -ForegroundColor Yellow
    try {
      $json = [Console]::In.ReadToEnd()
      $null = $json | ConvertFrom-Json
    } catch {
      Write-Error "Input doesn't look like valid JSON: $_"
      exit 1
    }
    Set-Content -Path $credFile -Value $json -NoNewline
    Write-Host "Wrote JSON content to $credFile" -ForegroundColor Green
    $secretsDir = Join-Path $repoRoot '.secrets'
    if (-not (Test-Path $secretsDir)) { New-Item -ItemType Directory -Path $secretsDir | Out-Null }
    $saJsonPath = Join-Path $secretsDir 'gcp_sa.json'
    Set-Content -Path $saJsonPath -Value $json -NoNewline
    Write-Host "Wrote JSON content to $saJsonPath" -ForegroundColor Green
  }
  '3' {
    # Need gcloud for this
    $gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
    if (-not $gcloud) {
      Write-Error "gcloud CLI not found. Install from https://cloud.google.com/sdk and try again, or use option 1/2."
      exit 1
    }

    $projectId = Read-Host "Enter your GCP PROJECT_ID (e.g., my-project-123)"
    if (-not $projectId) { Write-Error "PROJECT_ID is required."; exit 1 }
    $saId = Read-Host "Enter a service account ID (default: anchor-judge)"
    if (-not $saId) { $saId = 'anchor-judge' }
    $email = "$saId@$projectId.iam.gserviceaccount.com"
    $tempKey = Join-Path $env:TEMP "$saId.json"

    Write-Separator
    Write-Host "Enabling BigQuery API (if needed) ..." -ForegroundColor Cyan
    & gcloud services enable bigquery.googleapis.com --project "$projectId" 2>$null | Out-Null

    Write-Host "Creating service account (if needed) ..." -ForegroundColor Cyan
    & gcloud iam service-accounts create "$saId" --display-name "Anchor Judge" --project "$projectId" 2>$null | Out-Null

    Write-Host "Granting roles (Job User, Data Viewer) ..." -ForegroundColor Cyan
    & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$email" --role "roles/bigquery.jobUser" 2>$null | Out-Null
    & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$email" --role "roles/bigquery.dataViewer" 2>$null | Out-Null

    Write-Host "Creating key (temporary file) ..." -ForegroundColor Cyan
    if (Test-Path $tempKey) { Remove-Item $tempKey -Force -ErrorAction SilentlyContinue }
    & gcloud iam service-accounts keys create "$tempKey" --iam-account "$email" --project "$projectId" 2>$null | Out-Null

    if (-not (Test-Path $tempKey)) {
      Write-Error "Key creation failed: $tempKey not found."
      exit 1
    }

    try {
      $json = Get-Content -Raw -Path $tempKey
      $null = $json | ConvertFrom-Json
    } catch {
      Write-Error "Generated key doesn't look like valid JSON: $_"
      exit 1
    }

    # Validate JSON content is not empty
    if (-not $json -or $json.Trim() -eq '') {
      Write-Error "Generated key file is empty or contains no valid content."
      exit 1
    }

    Set-Content -Path $credFile -Value $json -NoNewline
    
    # Verify the file was written correctly
    if (-not (Test-Path $credFile) -or (Get-Item $credFile).Length -eq 0) {
      Write-Error "Failed to write content to $credFile - file is empty or doesn't exist"
      exit 1
    }

    # Clean up the temp key so we don't leave files lying around
    try { Remove-Item $tempKey -Force -ErrorAction SilentlyContinue } catch {}

    Write-Host "Wrote JSON content to $credFile (temporary key removed)" -ForegroundColor Green
    $secretsDir = Join-Path $repoRoot '.secrets'
    if (-not (Test-Path $secretsDir)) { New-Item -ItemType Directory -Path $secretsDir | Out-Null }
    $saJsonPath = Join-Path $secretsDir 'gcp_sa.json'
    Set-Content -Path $saJsonPath -Value $json -NoNewline
    
    # Verify the second file was written correctly too
    if (-not (Test-Path $saJsonPath) -or (Get-Item $saJsonPath).Length -eq 0) {
      Write-Error "Failed to write content to $saJsonPath - file is empty or doesn't exist"
      exit 1
    }
    
    Write-Host "Wrote JSON content to $saJsonPath" -ForegroundColor Green

    # Save these for IAM bootstrap defaults
    $createdProjectId = $projectId
    $createdSaEmail = $email
  }
  Default {
    Write-Error "Invalid choice. Run the script again and pick 1, 2, or 3."
    exit 1
  }
}

Write-Separator
if (Confirm-Or-Prompt "Run IAM bootstrap to grant minimal permissions for the demo?" $true) {
  # Check if we have the tools we need
  $gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
  $bq = Get-Command bq -ErrorAction SilentlyContinue
  if (-not $gcloud) {
    Write-Error "gcloud CLI not found. Install from https://cloud.google.com/sdk and try again."
    exit 1
  }
  if (-not $bq) {
    Write-Error "bq CLI not found. It comes with Cloud SDK; restart your shell after install."
    exit 1
  }

  # Get the info we need
  #  - Using PS 5.1-friendly prompts (no ternary operator)
  $projectPrompt = "Enter PROJECT_ID for IAM"
  if ($createdProjectId -ne '') { $projectPrompt += (" (default: {0})" -f $createdProjectId) }
  $projectId = Read-Host $projectPrompt
  if (-not $projectId) {
    if ($createdProjectId -ne '') { $projectId = $createdProjectId } else { Write-Error "PROJECT_ID is required"; exit 1 }
  }
  $region = Read-Host "Enter REGION for BigQuery/connection (default: US)"
  if (-not $region) { $region = 'US' }
  $datasetMain = Read-Host "Enter primary dataset (default: mskg_demo)"
  if (-not $datasetMain) { $datasetMain = 'mskg_demo' }
  $datasetStaging = Read-Host "Enter staging dataset (default: mskg_staging)"
  if (-not $datasetStaging) { $datasetStaging = 'mskg_staging' }
  $defaultSaEmail = if ($createdSaEmail -ne '') { $createdSaEmail } else { "anchor-judge@$projectId.iam.gserviceaccount.com" }
  $saPrompt = "Enter service account email to grant (default: $defaultSaEmail)"
  $saEmail = Read-Host $saPrompt
  if (-not $saEmail) { $saEmail = $defaultSaEmail }
  
  Write-Separator
  Write-Host "Signing into gcloud (if needed) and setting active project ..." -ForegroundColor Cyan
  try {
    $active = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null)
    if (-not $active) {
      Write-Host "No active account found. Opening login flow ..." -ForegroundColor Yellow
      & gcloud auth login --brief | Out-Null
    }
  } catch {}
  & gcloud config set project "$projectId" 2>$null | Out-Null

  Write-Host "Enabling required services ..." -ForegroundColor Cyan
  & gcloud services enable bigquery.googleapis.com bigqueryconnection.googleapis.com aiplatform.googleapis.com --project "$projectId" 2>$null | Out-Null

  Write-Host "Project-level roles ..." -ForegroundColor Cyan
  & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$saEmail" --role "roles/bigquery.user" 2>$null | Out-Null
  & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$saEmail" --role "roles/bigquery.jobUser" 2>$null | Out-Null
  & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$saEmail" --role "roles/bigquery.dataViewer" 2>$null | Out-Null

  # Optional broader roles for convenience in hackathon/sandbox contexts
  try {
    if (Confirm-Or-Prompt "Also grant BigQuery Admin to the service account (wider permissions)?" $false) {
      & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$saEmail" --role "roles/bigquery.admin" 2>$null | Out-Null
    }
    if (Confirm-Or-Prompt "Also grant PROJECT OWNER to the service account (full control; not recommended outside sandbox)?" $false) {
      & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$saEmail" --role "roles/owner" 2>$null | Out-Null
    }

    $activeUser = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null)
    if ($activeUser) {
      if (Confirm-Or-Prompt ("Grant BigQuery Admin to your active gcloud user?") $true) {
        & gcloud projects add-iam-policy-binding "$projectId" --member "user:$activeUser" --role "roles/bigquery.admin" 2>$null | Out-Null
      }
      if (Confirm-Or-Prompt ("Grant PROJECT OWNER to your active gcloud user? (use only if blocked)") $false) {
        & gcloud projects add-iam-policy-binding "$projectId" --member "user:$activeUser" --role "roles/owner" 2>$null | Out-Null
      }
    }

    # Vertex AI permissions for service account and (optionally) your user
    & gcloud projects add-iam-policy-binding "$projectId" --member "serviceAccount:$saEmail" --role "roles/aiplatform.user" 2>$null | Out-Null
    if ($activeUser) {
      if (Confirm-Or-Prompt ("Grant Vertex AI User to your active gcloud user?") $true) {
        & gcloud projects add-iam-policy-binding "$projectId" --member "user:$activeUser" --role "roles/aiplatform.user" 2>$null | Out-Null
      }
    }
  } catch {}

  Write-Host "Making sure datasets exist ..." -ForegroundColor Cyan
  try {
    $existing = (& bq --project_id=$projectId ls --datasets 2>$null)
    if (-not ($existing -match "\b$datasetMain\b")) {
      & bq --location=$region --project_id=$projectId mk -d "${projectId}:${datasetMain}" 2>$null | Out-Null
    }
    if (-not ($existing -match "\b$datasetStaging\b")) {
      & bq --location=$region --project_id=$projectId mk -d "${projectId}:${datasetStaging}" 2>$null | Out-Null
    }
  } catch {}

  # Align REGION to the primary dataset location (prevents AI connection region mismatch)
  try {
    $dsInfo = (& bq --project_id=$projectId --format=json show -d "${projectId}:${datasetMain}" 2>$null) | ConvertFrom-Json
    if ($dsInfo -and $dsInfo.location) { $region = $dsInfo.location }
  } catch {}

  Write-Host "Making sure BigQuery connection 'gemini' exists ..." -ForegroundColor Cyan
  try {
    $connList = (Get-Gcloud-BqConnOutput @('list','--project', "$projectId", '--location', "$region", '--format', 'value(name)'))
    $hasGemini = $false
    foreach ($c in $connList) { if ($c -match "/connections/gemini$") { $hasGemini = $true; break } }
    if (-not $hasGemini) {
      & bq mk --connection --location=$region --connection_type=CLOUD_RESOURCE --project_id=$projectId gemini 2>$null | Out-Null
    }
  } catch {}

  Write-Host "Granting Connection User on 'gemini' ..." -ForegroundColor Cyan
  $okGrantSa = Try-Gcloud-BqConn @('add-iam-policy-binding','--project', "$projectId", '--location', "$region", '--connection', 'gemini', '--member', "serviceAccount:$saEmail", '--role', 'roles/bigquery.connectionUser')
  if (-not $okGrantSa) { $null = Add-ConnIamMemberRest -ProjectId $projectId -Region $region -ConnectionId 'gemini' -Member "serviceAccount:$saEmail" -Role 'roles/bigquery.connectionUser' }
  try {
    $activeUser = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null)
    if ($activeUser) {
      $okGrantUser = Try-Gcloud-BqConn @('add-iam-policy-binding','--project', "$projectId", '--location', "$region", '--connection', 'gemini', '--member', "user:$activeUser", '--role', 'roles/bigquery.connectionUser')
      if (-not $okGrantUser) { $null = Add-ConnIamMemberRest -ProjectId $projectId -Region $region -ConnectionId 'gemini' -Member "user:$activeUser" -Role 'roles/bigquery.connectionUser' }
    }
  } catch {}

  Write-Host "Granting Vertex AI User to BigQuery service agent ..." -ForegroundColor Cyan
  try {
    $pn = (& gcloud projects describe $projectId --format='value(projectNumber)' 2>$null)
    if ($pn) {
      try { & gcloud projects add-iam-policy-binding $projectId --member "serviceAccount:service-$pn@gcp-sa-bigquery.iam.gserviceaccount.com" --role "roles/aiplatform.user" 2>$null | Out-Null } catch {}
      try {
        $okGrantBqAgent = Try-Gcloud-BqConn @('add-iam-policy-binding','--project', "$projectId", '--location', "$region", '--connection', 'gemini', '--member', "serviceAccount:service-$pn@gcp-sa-bigquery.iam.gserviceaccount.com", '--role', 'roles/bigquery.connectionUser')
        if (-not $okGrantBqAgent) { $null = Add-ConnIamMemberRest -ProjectId $projectId -Region $region -ConnectionId 'gemini' -Member "serviceAccount:service-$pn@gcp-sa-bigquery.iam.gserviceaccount.com" -Role 'roles/bigquery.connectionUser' }
      } catch {}
    }
  } catch {}

  # Grant helpful roles to the connection service identity (bqcx-...condel) if present
  try {
    $cxSa = (Get-Gcloud-BqConnOutput @('describe','--project', "$projectId", '--location', "$region", 'gemini', '--format', 'value(connection.serviceAccountId)'))
    if ($cxSa) {
      & gcloud projects add-iam-policy-binding $projectId --member "serviceAccount:$cxSa" --role "roles/aiplatform.user" 2>$null | Out-Null
      & gcloud projects add-iam-policy-binding $projectId --member "serviceAccount:$cxSa" --role "roles/storage.objectViewer" 2>$null | Out-Null
    }
  } catch {}

  Write-Separator
  Write-Host "IAM bootstrap complete." -ForegroundColor Green
  Write-Host "Project: $projectId" -ForegroundColor Gray
  Write-Host "Region:  $region" -ForegroundColor Gray
  Write-Host "Datasets: $datasetMain, $datasetStaging" -ForegroundColor Gray
  Write-Host "Service Account: $saEmail" -ForegroundColor Gray

  # Write a .env so the notebook picks the right connection automatically
  try {
    $envPath = Join-Path $repoRoot '.env'
    $envLines = @(
      "PROJECT_ID=$projectId",
      "REGION=$region",
      "DATASET_ID=$datasetMain",
      "STAGING_DATASET_ID=$datasetStaging",
      "BQ_CONNECTION_NAME=$projectId.$region.gemini",
      "QUICKSTART=true"
    )
    Set-Content -Path $envPath -Value ($envLines -join "`n") -NoNewline
    Write-Host "Wrote .env with BQ_CONNECTION_NAME=$projectId.$region.gemini" -ForegroundColor Green
  } catch {}

  Write-Separator
  Write-Host "Next: Restart the notebook kernel, re-run the first cell. You should see 'AI connection OK'." -ForegroundColor Cyan
}
else {
  Write-Host "Skipped IAM bootstrap. You can re-run this script later to apply it." -ForegroundColor Yellow
}

Write-Separator
Write-Host "Done." -ForegroundColor Green
Write-Separator 