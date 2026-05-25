Add-Type -AssemblyName System.Drawing

$out = Join-Path $PSScriptRoot '..\assets\images\og-default.png'
$out = [System.IO.Path]::GetFullPath($out)

$W = 1200
$H = 630

$bmp = New-Object System.Drawing.Bitmap($W, $H)
$g   = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode     = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic

# --- background: vertical dark gradient ---
$bgRect = New-Object System.Drawing.Rectangle(0, 0, $W, $H)
$bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    $bgRect,
    [System.Drawing.Color]::FromArgb(255, 6, 8, 15),
    [System.Drawing.Color]::FromArgb(255, 13, 21, 32),
    [System.Drawing.Drawing2D.LinearGradientMode]::Vertical
)
$g.FillRectangle($bgBrush, $bgRect)
$bgBrush.Dispose()

# --- top-left blue orb ---
$orb1 = New-Object System.Drawing.Drawing2D.GraphicsPath
$orb1.AddEllipse(-200, -200, 700, 700)
$pgb1 = New-Object System.Drawing.Drawing2D.PathGradientBrush($orb1)
$pgb1.CenterColor = [System.Drawing.Color]::FromArgb(110, 56, 189, 248)
$pgb1.SurroundColors = ,[System.Drawing.Color]::FromArgb(0, 56, 189, 248)
$g.FillPath($pgb1, $orb1)
$pgb1.Dispose(); $orb1.Dispose()

# --- bottom-right purple orb ---
$orb2 = New-Object System.Drawing.Drawing2D.GraphicsPath
$orb2.AddEllipse(700, 250, 700, 700)
$pgb2 = New-Object System.Drawing.Drawing2D.PathGradientBrush($orb2)
$pgb2.CenterColor = [System.Drawing.Color]::FromArgb(110, 192, 132, 252)
$pgb2.SurroundColors = ,[System.Drawing.Color]::FromArgb(0, 192, 132, 252)
$g.FillPath($pgb2, $orb2)
$pgb2.Dispose(); $orb2.Dispose()

# --- thin top accent line ---
$accentBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    (New-Object System.Drawing.Rectangle(0, 0, $W, 4)),
    [System.Drawing.Color]::FromArgb(255, 56, 189, 248),
    [System.Drawing.Color]::FromArgb(255, 192, 132, 252),
    [System.Drawing.Drawing2D.LinearGradientMode]::Horizontal
)
$g.FillRectangle($accentBrush, 0, 0, $W, 4)
$accentBrush.Dispose()

# --- brand dot + name (top-left) ---
$dotBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    (New-Object System.Drawing.Rectangle(72, 70, 28, 28)),
    [System.Drawing.Color]::FromArgb(255, 56, 189, 248),
    [System.Drawing.Color]::FromArgb(255, 192, 132, 252),
    [System.Drawing.Drawing2D.LinearGradientMode]::ForwardDiagonal
)
$g.FillEllipse($dotBrush, 72, 70, 28, 28)
$dotBrush.Dispose()

$brandFont = New-Object System.Drawing.Font('Segoe UI', 22, [System.Drawing.FontStyle]::Bold)
$brandBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 240, 248, 255))
$g.DrawString('DarryPy', $brandFont, $brandBrush, 110, 70)
$brandFont.Dispose(); $brandBrush.Dispose()

# --- main title (large, center-left) ---
$titleFont = New-Object System.Drawing.Font('Microsoft YaHei', 60, [System.Drawing.FontStyle]::Bold)
$titleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 240, 248, 255))
$g.DrawString('DarryPy 的个人博客', $titleFont, $titleBrush, 72, 220)
$titleFont.Dispose(); $titleBrush.Dispose()

# --- subtitle gradient ---
$subFont = New-Object System.Drawing.Font('Microsoft YaHei', 34, [System.Drawing.FontStyle]::Regular)
$subRect = New-Object System.Drawing.Rectangle(72, 330, 1080, 60)
$subBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    $subRect,
    [System.Drawing.Color]::FromArgb(255, 125, 211, 252),
    [System.Drawing.Color]::FromArgb(255, 192, 132, 252),
    [System.Drawing.Drawing2D.LinearGradientMode]::Horizontal
)
$g.DrawString('后端  ·  系统  ·  AI 工具', $subFont, $subBrush, 72, 330)
$subFont.Dispose(); $subBrush.Dispose()

# --- tag pill row ---
$tagFont = New-Object System.Drawing.Font('Segoe UI', 16, [System.Drawing.FontStyle]::Bold)
$tags = @('Prompt', 'RAG', 'Agent', 'Eval & Safety', '工程实战')
$x = 72
$y = 470
$pad = 18
$gap = 14
foreach ($t in $tags) {
    $sz = $g.MeasureString($t, $tagFont)
    $w = [int]$sz.Width + $pad * 2
    $h = 40
    $r = New-Object System.Drawing.Rectangle($x, $y, $w, $h)
    $bg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(40, 56, 189, 248))
    $g.FillRectangle($bg, $r)
    $bg.Dispose()
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(120, 56, 189, 248), 1)
    $g.DrawRectangle($pen, $r)
    $pen.Dispose()
    $tb = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 125, 211, 252))
    $g.DrawString($t, $tagFont, $tb, $x + $pad, $y + 8)
    $tb.Dispose()
    $x += $w + $gap
}
$tagFont.Dispose()

# --- footer url ---
$urlFont = New-Object System.Drawing.Font('Segoe UI', 18, [System.Drawing.FontStyle]::Regular)
$urlBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 91, 118, 147))
$g.DrawString('darrypy.github.io', $urlFont, $urlBrush, 72, 555)
$urlFont.Dispose(); $urlBrush.Dispose()

$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()

Write-Output "wrote $out"
