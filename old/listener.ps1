Add-Type -AssemblyName System.Speech

$uv = (Get-Command uv.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$pilotScript = Join-Path $PSScriptRoot 'pilot.py'
$pilotProcess = $null

$installed = @(
    [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
)

if ($installed.Count -eq 0) {
    throw "System.Speech could not find a compatible recognizer in this PowerShell session."
}

$listener = [System.Speech.Recognition.SpeechRecognitionEngine]::new(
    $installed[0]
)

try {
    # These are the phrases the listener is allowed to recognize.
    $phrases = [System.Speech.Recognition.Choices]::new()
    $phrases.Add([string[]]@("pilot", "stop listener"))

    $builder = [System.Speech.Recognition.GrammarBuilder]::new()
    $builder.Culture = $listener.RecognizerInfo.Culture
    $builder.Append($phrases)

    $grammar = [System.Speech.Recognition.Grammar]::new($builder)
    $listener.LoadGrammar($grammar)
    $listener.SetInputToDefaultAudioDevice()

    Write-Host 'Listening. Say "pilot" to launch Pilot, or "stop listener" to stop listening.'
    $running = $true

    while ($running) {
        $result = $listener.Recognize([TimeSpan]::FromSeconds(3))

        # Ignore silence and uncertain matches.
        if ($null -eq $result -or $result.Confidence -lt 0.50) {
            continue
        }

        Write-Host "Heard: $($result.Text)"

        switch ($result.Text) {
            "pilot" {
                if ($null -ne $pilotProcess -and -not $pilotProcess.HasExited) {
                    Write-Host 'Pilot is already running.'
                }
                else {
                    try {
                        $pilotProcess = Start-Process -FilePath $uv -ArgumentList @(
                            'run', '--project', "`"$PSScriptRoot`"", 'python', "`"$pilotScript`""
                        ) -WorkingDirectory $PSScriptRoot -PassThru -ErrorAction Stop
                        Write-Host 'Started Pilot.'
                    }
                    catch {
                        Write-Warning "Could not start Pilot: $_"
                    }
                }
            }
            "stop listener" {
                $running = $false
            }
        }
    }
}
finally {
    $listener.Dispose()
}
