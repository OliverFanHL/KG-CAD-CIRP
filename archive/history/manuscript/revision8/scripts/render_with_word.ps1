param([string]$DocumentRoot)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $DocumentRoot).Path
$reviewRoot = Join-Path $taskRoot '.qa'
New-Item -ItemType Directory -Force -Path $reviewRoot | Out-Null
$wordApp = $null
$document = $null
$records = @()
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordApp.AutomationSecurity = 3
    foreach ($stem in @('manuscript','supplement')) {
        $inputFile = Join-Path $taskRoot ($stem + '.docx')
        $renderFolder = Join-Path $reviewRoot $stem
        New-Item -ItemType Directory -Force -Path $renderFolder | Out-Null
        $outputFile = Join-Path $renderFolder ($stem + '.pdf')
        $document = $wordApp.Documents.Open($inputFile, $false, $false, $false)
        $document.UpdateStylesOnOpen = $false
        $document.Fields.Update() | Out-Null
        $document.Repaginate()
        $pages = $document.ComputeStatistics(2)
        $document.Save()
        $document.ExportAsFixedFormat($outputFile,17)
        $records += [pscustomobject]@{document=$stem; pages=$pages; equations=$document.OMaths.Count; tables=$document.Tables.Count}
        $document.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
        $document = $null
    }
    $records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $reviewRoot 'word_render.json') -Encoding utf8
    $records | ConvertTo-Json -Depth 4
}
finally {
    if ($null -ne $document) { $document.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($null -ne $wordApp) { $wordApp.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) }
}
