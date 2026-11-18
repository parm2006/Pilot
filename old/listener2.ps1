# listener.ps1 - fully local voice commands (Windows System.Speech, no network, no API)
#
# Say a phrase from commands.json and it runs. commands.json is re-read automatically
# whenever you save it, so you can add or change commands while the listener is running.
#
# Each command does ONE of these:
#   "keys":    "ctrl+c"                      press a key combo
#              ["home", "shift+end"]         press several combos in order
#   "text":    "you@example.com"             type this text
#   "run":     "notepad.exe"                 launch a program
#   "args":    "..."                         (optional) arguments for "run"
#   "builtin": "pilot" | "pause" | "resume" | "quit"
#
# Key names: a-z, 0-9, f1-f24, enter, tab, esc, space, backspace, delete, insert, home, end,
#            pageup, pagedown, up, down, left, right, printscreen, capslock
# Modifiers: ctrl, alt, shift, win          (join with +, e.g. "ctrl+shift+t")
#
# NOTE: Windows blocks keystrokes from a normal process into an *elevated* (admin) window.
#       Run this script as administrator if you need to control those.

Add-Type -AssemblyName System.Speech
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Kbd {
    [DllImport("user32.dll")]
    public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
}
'@

$ConfigPath   = Join-Path $PSScriptRoot 'commands.json'
$pilotScript  = Join-Path $PSScriptRoot 'pilot.py'
$pilotProcess = $null

# ----------------------------------------------------------------------------- keys
$Modifiers = @{ ctrl = 0x11; control = 0x11; alt = 0x12; shift = 0x10; win = 0x5B; windows = 0x5B }
$Named = @{
    enter = 0x0D; return = 0x0D; tab = 0x09; esc = 0x1B; escape = 0x1B; space = 0x20
    backspace = 0x08; delete = 0x2E; del = 0x2E; insert = 0x2D; home = 0x24; end = 0x23
    pageup = 0x21; pagedown = 0x22; left = 0x25; up = 0x26; right = 0x27; down = 0x28
    printscreen = 0x2C; capslock = 0x14
}
# Keys that need the "extended key" flag so Windows treats them as the navigation keys.
$Extended = @(0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E)

function Get-VK([string]$name) {
    $n = $name.Trim().ToLowerInvariant()
    if ($Modifiers.ContainsKey($n)) { return [int]$Modifiers[$n] }
    if ($Named.ContainsKey($n))     { return [int]$Named[$n] }
    if ($n -match '^[a-z]$')        { return [int][char]$n.ToUpperInvariant() }
    if ($n -match '^[0-9]$')        { return [int][char]$n }
    if ($n -match '^f([1-9]|1[0-9]|2[0-4])$') { return 0x6F + [int]$Matches[1] }
    throw "Unknown key '$name'"
}

function Send-VK([int]$vk, [bool]$up) {
    $flags = 0
    if ($up) { $flags = 2 }
    if ($Extended -contains $vk) { $flags = $flags -bor 1 }
    [Kbd]::keybd_event([byte]$vk, 0, [uint32]$flags, [UIntPtr]::Zero)
}

function Send-Combo([string]$combo, [int]$delayMs) {
    # Resolve every key first so a typo can never leave a modifier stuck down.
    $vks = @($combo -split '\+' | ForEach-Object { Get-VK $_ })
    foreach ($vk in $vks) { Send-VK $vk $false; Start-Sleep -Milliseconds $delayMs }
    [array]::Reverse($vks)
    foreach ($vk in $vks) { Send-VK $vk $true;  Start-Sleep -Milliseconds $delayMs }
}

function Send-Text([string]$text) {
    # SendKeys treats + ^ % ~ ( ) { } [ ] as special; wrap each in braces to type it literally.
    $escaped = [regex]::Replace($text, '[+^%~(){}\[\]]', { param($m) '{' + $m.Value + '}' })
    [System.Windows.Forms.SendKeys]::SendWait($escaped)
}

# ----------------------------------------------------------------------------- config
$DefaultConfig = @'
{
  "settings": { "minConfidence": 0.60, "keyDelayMs": 15 },
  "commands": {
    "copy": { "keys": "ctrl+c" },
    "cut": { "keys": "ctrl+x" },
    "paste": { "keys": "ctrl+v" },
    "select all": { "keys": "ctrl+a" },
    "pilot": { "builtin": "pilot" },
    "stop listening": { "builtin": "pause" },
    "start listening": { "builtin": "resume" },
    "stop listener": { "builtin": "quit" }
  }
}
'@
if (-not (Test-Path $ConfigPath)) { Set-Content -Path $ConfigPath -Value $DefaultConfig -Encoding UTF8 }

function Read-Config {
    $cfg = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
    $table = @{}
    foreach ($p in $cfg.commands.PSObject.Properties) {
        $phrase = $p.Name.Trim().ToLowerInvariant()
        if ($phrase) { $table[$phrase] = $p.Value }
    }
    if ($table.Count -eq 0) { throw 'commands.json has no commands.' }
    $min = 0.60; $delay = 15
    if ($null -ne $cfg.settings) {
        if ($null -ne $cfg.settings.minConfidence) { $min   = [double]$cfg.settings.minConfidence }
        if ($null -ne $cfg.settings.keyDelayMs)    { $delay = [int]$cfg.settings.keyDelayMs }
    }
    return @{ Commands = $table; MinConfidence = $min; KeyDelayMs = $delay }
}

function Load-Commands($listener, $config) {
    $choices = [System.Speech.Recognition.Choices]::new()
    $choices.Add([string[]]@($config.Commands.Keys))
    $builder = [System.Speech.Recognition.GrammarBuilder]::new()
    $builder.Culture = $listener.RecognizerInfo.Culture
    $builder.Append($choices)
    $listener.UnloadAllGrammars()
    $listener.LoadGrammar([System.Speech.Recognition.Grammar]::new($builder))
}

# ----------------------------------------------------------------------------- actions
function Start-Pilot {
    if ($null -ne $script:pilotProcess -and -not $script:pilotProcess.HasExited) {
        Write-Host 'Pilot is already running.'
        return
    }
    try {
        $uv = (Get-Command uv.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
        $script:pilotProcess = Start-Process -FilePath $uv -ArgumentList @(
            'run', '--project', "`"$PSScriptRoot`"", 'python', "`"$pilotScript`""
        ) -WorkingDirectory $PSScriptRoot -PassThru -ErrorAction Stop
        Write-Host 'Started Pilot.'
    }
    catch { Write-Warning "Could not start Pilot: $_" }
}

# Returns 'pause' / 'resume' / 'quit' for builtins that change listener state, else $null.
function Invoke-Command($action, $config) {
    if ($null -ne $action.builtin) {
        switch ($action.builtin) {
            'pilot'  { Start-Pilot; return $null }
            'pause'  { return 'pause' }
            'resume' { return 'resume' }
            'quit'   { return 'quit' }
            default  { Write-Warning "Unknown builtin '$($action.builtin)'" }
        }
        return $null
    }
    if ($null -ne $action.keys) {
        foreach ($combo in @($action.keys)) { Send-Combo ([string]$combo) $config.KeyDelayMs }
    }
    if ($null -ne $action.text) { Send-Text ([string]$action.text) }
    if ($null -ne $action.run) {
        if ($null -ne $action.args) { Start-Process -FilePath ([string]$action.run) -ArgumentList ([string]$action.args) }
        else                        { Start-Process -FilePath ([string]$action.run) }
    }
    return $null
}

# ----------------------------------------------------------------------------- main
$installed = @([System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers())
if ($installed.Count -eq 0) {
    throw 'System.Speech could not find a compatible recognizer in this PowerShell session.'
}
$listener = [System.Speech.Recognition.SpeechRecognitionEngine]::new($installed[0])

try {
    $config     = Read-Config
    $configTime = (Get-Item $ConfigPath).LastWriteTimeUtc
    Load-Commands $listener $config
    $listener.SetInputToDefaultAudioDevice()

    Write-Host "Listening for $($config.Commands.Count) commands (edit commands.json to change them)."
    Write-Host 'Say "stop listening" to pause, "start listening" to resume, "stop listener" to quit.'
    $running = $true
    $paused  = $false

    while ($running) {
        # Hot-reload commands.json when it changes; keep the old set if the new one is broken.
        $stamp = (Get-Item $ConfigPath).LastWriteTimeUtc
        if ($stamp -ne $configTime) {
            $configTime = $stamp
            try {
                $config = Read-Config
                Load-Commands $listener $config
                Write-Host "Reloaded commands.json ($($config.Commands.Count) commands)."
            }
            catch { Write-Warning "commands.json not reloaded: $_" }
        }

        $result = $listener.Recognize([TimeSpan]::FromSeconds(3))
        if ($null -eq $result -or $result.Confidence -lt $config.MinConfidence) { continue }

        $phrase = $result.Text.ToLowerInvariant()
        $action = $config.Commands[$phrase]
        if ($null -eq $action) { continue }

        # While paused, only the resume / quit builtins are honoured.
        if ($paused -and $action.builtin -notin @('resume', 'quit')) { continue }

        Write-Host ("Heard: {0} ({1:P0})" -f $phrase, $result.Confidence)
        try { $state = Invoke-Command $action $config }
        catch { Write-Warning "Command '$phrase' failed: $_"; continue }

        switch ($state) {
            'pause'  { $paused = $true;  Write-Host 'Paused.' }
            'resume' { $paused = $false; Write-Host 'Listening again.' }
            'quit'   { $running = $false }
        }
    }
}
finally {
    $listener.Dispose()
}
