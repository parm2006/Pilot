[CmdletBinding()]
param([switch]$Check, [switch]$Registered)

$ErrorActionPreference = 'Stop'
$wakePhrases = @('pilot', 'start pilot', 'start listener')
$ignoredPhrases = @('stop', 'stop pilot', 'stop listener', 'quit pilot', 'pause pilot', 'autopilot', 'pilot light')
$runKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'

function Get-EnglishRecognizer {
    Add-Type -AssemblyName System.Speech
    $recognizers = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
    $recognizer = $recognizers | Where-Object { $_.Culture.Name -eq 'en-US' } | Select-Object -First 1
    if (-not $recognizer) {
        $recognizer = $recognizers | Where-Object {
            $_.Culture.TwoLetterISOLanguageName -eq 'en'
        } | Select-Object -First 1
    }
    if (-not $recognizer) {
        throw 'Install English Speech in Windows Settings to use the Pilot wake listener.'
    }
    return $recognizer
}

function New-WakeEngine($Recognizer) {
    $engine = [System.Speech.Recognition.SpeechRecognitionEngine]::new($Recognizer)
    try {
        $choices = [System.Speech.Recognition.Choices]::new()
        $choices.Add([string[]]($wakePhrases + $ignoredPhrases))
        $builder = [System.Speech.Recognition.GrammarBuilder]::new()
        $builder.Culture = $Recognizer.Culture
        $builder.Append($choices)
        $engine.LoadGrammar([System.Speech.Recognition.Grammar]::new($builder))
        return $engine
    } catch {
        $engine.Dispose()
        throw
    }
}

function Enter-Mutex($Mutex) {
    try { return $Mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { return $true }
}

function Test-PilotOpen($Desktop) {
    if (-not (Enter-Mutex $Desktop)) { return $true }
    $Desktop.ReleaseMutex()
    return $false
}

function Test-WakeResult($Result) {
    return $null -ne $Result -and $Result.Confidence -ge 0.72 -and $Result.Text -in $wakePhrases
}

function Wait-WakePhrase($Recognizer, $Desktop, $StopSignal) {
    $engine = New-WakeEngine $Recognizer
    $eventName = 'Pilot.WakeRecognition'
    try {
        $engine.SetInputToDefaultAudioDevice()
        Register-ObjectEvent -InputObject $engine -EventName SpeechRecognized -SourceIdentifier $eventName | Out-Null
        $engine.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)
        while (-not $StopSignal.WaitOne(0) -and -not (Test-PilotOpen $Desktop)) {
            $heard = Wait-Event -SourceIdentifier $eventName -Timeout 1
            if ($null -ne $heard) {
                Remove-Event -EventIdentifier $heard.EventIdentifier
                if (Test-WakeResult $heard.SourceEventArgs.Result) { return $true }
            }
        }
        return $false
    } finally {
        Unregister-Event -SourceIdentifier $eventName -ErrorAction SilentlyContinue
        Remove-Event -SourceIdentifier $eventName -ErrorAction SilentlyContinue
        $engine.Dispose()
    }
}

function Start-PilotDesktop {
    $appFolder = Split-Path -Parent $PSScriptRoot
    $pilotExe = Join-Path $appFolder 'Pilot.exe'
    if ((Split-Path -Leaf $PSScriptRoot) -eq '_internal' -and (Test-Path -LiteralPath $pilotExe)) {
        return Start-Process -FilePath $pilotExe -WorkingDirectory $appFolder -WindowStyle Normal -PassThru
    }
    $python = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
    $app = Join-Path $PSScriptRoot 'pilot_app.py'
    return Start-Process -FilePath $python -ArgumentList ('"' + $app + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Normal -PassThru
}

# Keep the recognizer and its microphone input idle while the desktop app owns the session.
function Invoke-PilotWake($Recognizer) {
    $wake = [System.Threading.Mutex]::new($false, 'Local\Pilot.Wake')
    $desktop = [System.Threading.Mutex]::new($false, 'Local\Pilot.Desktop')
    $microphone = [System.Threading.Mutex]::new($false, 'Local\Pilot.WakeMicrophone')
    $stopSignal = [System.Threading.EventWaitHandle]::new($false, 'ManualReset', 'Local\Pilot.WakeStop')
    $ownsWake = Enter-Mutex $wake
    try {
        if (-not $ownsWake) { return }
        while (-not $stopSignal.WaitOne(0)) {
            if ($Registered -and -not (Get-ItemPropertyValue -Path $runKey -Name PilotWakeListener -ErrorAction SilentlyContinue)) {
                return
            }
            if ((Test-PilotOpen $desktop) -or -not (Enter-Mutex $microphone)) {
                $null = $stopSignal.WaitOne(350)
                continue
            }
            try {
                $launch = -not (Test-PilotOpen $desktop) -and (Wait-WakePhrase $Recognizer $desktop $stopSignal)
            } catch {
                Write-Warning $_
                $launch = $false
                $null = $stopSignal.WaitOne(3000)
            } finally {
                $microphone.ReleaseMutex()
            }
            if ($launch -and -not $stopSignal.WaitOne(0) -and -not (Test-PilotOpen $desktop)) {
                try {
                    $pilot = Start-PilotDesktop
                    try {
                        while (-not $pilot.HasExited -and -not $stopSignal.WaitOne(350)) {}
                    } finally {
                        $pilot.Dispose()
                    }
                } catch {
                    Write-Warning $_
                    $null = $stopSignal.WaitOne(3000)
                }
            }
        }
    } finally {
        if ($ownsWake) { $wake.ReleaseMutex() }
        $stopSignal.Dispose()
        $microphone.Dispose()
        $desktop.Dispose()
        $wake.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $recognizer = Get-EnglishRecognizer
        if ($Check) { Write-Output $recognizer.Culture.Name }
        else { Invoke-PilotWake $recognizer }
    } catch {
        Write-Error $_
        exit 1
    }
}
