# Opens the report in Microsoft Word, refreshes the table of contents and page numbers,
# saves the .docx and exports a print-ready PDF next to it.
# Usage: powershell -ExecutionPolicy Bypass -File report\export_pdf.ps1
$docx = Get-ChildItem $PSScriptRoot -Filter *.docx | Where-Object Name -notlike '~$*' | Select-Object -First 1
$pdf = [IO.Path]::ChangeExtension($docx.FullName, '.pdf')
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx.FullName)
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)  # 17 = wdExportFormatPDF
    "pages: $($doc.ComputeStatistics(2))"
    $doc.Close()
    "saved $pdf"
} finally {
    $word.Quit()
}
