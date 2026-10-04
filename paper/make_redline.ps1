# Marked-up copy for the advisor: Word "Compare" of the advisor's ver.2 (all tracked changes accepted)
# against the rebuilt draft, so every edit since ver.2 shows as a tracked change (insertions/deletions).
#   powershell -File paper/make_redline.ps1 <ver2_accepted.docx> <new.docx> <out_redline.docx>
param([string]$Old, [string]$New, [string]$Out)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $a = $w.Documents.Open((Resolve-Path $Old).Path, $false, $true)
    $b = $w.Documents.Open((Resolve-Path $New).Path, $false, $true)
    # wdCompareDestinationNew = 2, wdGranularityWordLevel = 1; keep formatting changes out (only text)
    $r = $w.CompareDocuments($a, $b, 2, 1, $true, $true, $false, $true, $true, $true, $true, $true, $true, $true, "수정 (2026-10-04)", $true)
    "revisions: " + $r.Revisions.Count
    $r.SaveAs2([System.IO.Path]::GetFullPath($Out), 16)
    $pdf = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetFullPath($Out), ".pdf")
    $r.ActiveWindow.View.RevisionsFilter.Markup = 2          # all markup
    $r.ActiveWindow.View.RevisionsFilter.View = 0            # final
    $r.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 1, 1, 7)  # wdExportDocumentWithMarkup = 7
    $r.Close(0); $a.Close(0); $b.Close(0)
} finally { $w.Quit() }
