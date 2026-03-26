# MLflow tracking server startup script for Windows (PowerShell)
# Usage: .\mlflow_server.ps1

# Load .env from project root
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
        $key, $val = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($key.Trim(), $val.Trim(), 'Process')
    }
}

$user     = if ($env:MLFLOW_POSTGRES_USER)     { $env:MLFLOW_POSTGRES_USER }     else { "postgres" }
$password = if ($env:MLFLOW_POSTGRES_PASSWORD) { $env:MLFLOW_POSTGRES_PASSWORD } else { "" }
$db       = if ($env:MLFLOW_POSTGRES_DB)       { $env:MLFLOW_POSTGRES_DB }       else { "mlflow_db" }
$port     = if ($env:MLFLOW_PORT)              { $env:MLFLOW_PORT }              else { "5001" }

if ($password) {
    $dbUri = "postgresql+psycopg2://${user}:${password}@localhost:5432/${db}"
} else {
    $dbUri = "postgresql+psycopg2://${user}@localhost:5432/${db}"
}

mlflow server `
  --backend-store-uri $dbUri `
  --default-artifact-root ./mlruns `
  --host 0.0.0.0 `
  --port $port
