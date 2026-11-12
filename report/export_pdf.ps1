# Refresh the TOC, apply final typography and links, then export the print PDF.
param([string]$PythonExe = "python")
$ReportDocx = Join-Path $PSScriptRoot 'YOLO-Master_PPE_Course_Design.docx'
$ReportPdf = [IO.Path]::ChangeExtension($ReportDocx,'.pdf')
$ReportWord = New-Object -ComObject Word.Application
$ReportWord.Visible = $false
$ReportWord.DisplayAlerts = 0
try {
    $ReportDocument = $ReportWord.Documents.Open($ReportDocx)
    $ReportDocument.Repaginate()
    foreach ($ReportToc in $ReportDocument.TablesOfContents) {
        $ReportToc.Update()
        $ReportToc.TabLeader = 1 # wdTabLeaderDots
        $ReportToc.Range.ParagraphFormat.TabStops.Add(453.5,2,1) | Out-Null
    }
    $ReportDocument.Fields.Update() | Out-Null
    $ReportDocument.Save()
    $ReportDocument.Close(0)
} finally { $ReportWord.Quit() }
& $PythonExe (Join-Path $PSScriptRoot 'final_format_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Final report formatting failed.' }
$ReportWord = New-Object -ComObject Word.Application
$ReportWord.Visible = $false
$ReportWord.DisplayAlerts = 0
try {
    $ReportDocument = $ReportWord.Documents.Open($ReportDocx,$false,$true)
    $ReportDocument.ExportAsFixedFormat($ReportPdf,17)
    Write-Output "pages: $($ReportDocument.ComputeStatistics(2))"
    $ReportDocument.Close(0)
    Write-Output "saved $ReportPdf"
} finally { $ReportWord.Quit() }
