; Finetuner Inno Setup script
; Native x64, native ARM64, or a combined installer that picks files at install time.
;
;   iscc /DAppArch=x64 /DAppVersion=0.2.0 /DDistFolder=..\dist\Finetuner-x64 scripts\finetuner_installer.iss
;   iscc /DAppArch=arm64 /DAppVersion=0.2.0 /DDistFolder=..\dist\Finetuner-arm64 scripts\finetuner_installer.iss
;   iscc /DAppArch=universal /DAppVersion=0.2.0 scripts\finetuner_installer.iss

#if Ver < EncodeVer(6,3,0,0)
  #error Inno Setup 6.3 or newer is required (x64os + ARM64).
#endif

#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
#ifndef AppArch
  #define AppArch "x64"
#endif
#ifndef DistFolder
  #if AppArch == "arm64"
    #define DistFolder "..\dist\Finetuner-arm64"
  #else
    #define DistFolder "..\dist\Finetuner-x64"
  #endif
#endif

#define AppName "Finetuner"
#define AppPublisher "Finetuner"
#define AppExe "Finetuner.exe"
#define AppURL "https://github.com"

#if AppArch == "universal"
  #define SetupSuffix "universal"
  #define AllowedArchs "x64os arm64"
#elif AppArch == "arm64"
  #define SetupSuffix "arm64"
  #define AllowedArchs "arm64"
#else
  #define SetupSuffix "x64"
  #define AllowedArchs "x64os"
#endif

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} ({#SetupSuffix})
AppPublisher={#AppPublisher}
AppMutex=Finetuner.Finetuner.1
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=Finetuner-Setup-{#SetupSuffix}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed={#AllowedArchs}
ArchitecturesInstallIn64BitMode=x64os arm64
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
CloseApplications=yes
UninstallDisplayIcon={app}\{#AppExe}
#ifexist "..\assets\icon.ico"
SetupIconFile=..\assets\icon.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
#if AppArch == "universal"
Source: "..\dist\Finetuner-x64\*"; DestDir: "{app}"; Check: IsX64OS; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\Finetuner-arm64\*"; DestDir: "{app}"; Check: IsArm64; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "{#DistFolder}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; Comment: "Local LLM post-training workbench"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\*.pyc"

[Messages]
WindowsVersionNotSupported=Finetuner needs 64-bit Windows 10 or 11 (x64 or ARM64).
OnlyOnTheseArchitectures=This installer is for {#SetupSuffix} Windows. Download Finetuner-Setup-x64.exe or Finetuner-Setup-arm64.exe for this PC, or the universal installer.
