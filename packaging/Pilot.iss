#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId=PilotDesktop
AppName=Pilot
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\Pilot
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
WizardStyle=modern
OutputDir=..\dist\windows-release
OutputBaseFilename=PilotSetup
UninstallDisplayIcon={app}\Pilot.exe
Compression=lzma2/fast
SolidCompression=yes

[Files]
Source: "..\dist\Pilot\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\Pilot"; Filename: "{app}\Pilot.exe"; WorkingDir: "{app}"
Name: "{userprograms}\Pilot Wake Listener Settings"; Filename: "{app}\Pilot.exe"; Parameters: "--wake-settings"; WorkingDir: "{app}"

[Run]
Filename: "{app}\Pilot.exe"; Description: "Open Pilot"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "{app}\Pilot.exe"; Parameters: "--disable-wake"; Flags: runhidden; RunOnceId: "DisablePilotWake"
