@echo off
rem grant-lock-pages.bat - give a Windows account the "Lock pages in memory" right (SeLockMemoryPrivilege),
rem which Basalt needs for 2 MB large pages. Run it as administrator (right-click > Run as administrator).
rem   grant-lock-pages.bat            grants it to the account that started the script
rem   grant-lock-pages.bat NAME       grants it to the account NAME (e.g. sergi or PC\sergi)
rem The right applies after that account signs out and back in.
setlocal EnableExtensions EnableDelayedExpansion

net session >nul 2>&1
if errorlevel 1 (
    echo This script must run as administrator: right-click it and choose "Run as administrator".
    pause
    exit /b 1
)

set "ACCOUNT=%~1"
if "%ACCOUNT%"=="" set "ACCOUNT=%USERDOMAIN%\%USERNAME%"

rem Resolve the account to its SID, the form the policy file uses.
set "SID="
for /f "usebackq delims=" %%s in (`powershell -NoProfile -Command "try { (New-Object System.Security.Principal.NTAccount('%ACCOUNT%')).Translate([System.Security.Principal.SecurityIdentifier]).Value } catch { '' }"`) do set "SID=%%s"
if "%SID%"=="" (
    echo Account "%ACCOUNT%" not found.
    pause
    exit /b 1
)
echo Account: %ACCOUNT%  ^(%SID%^)

set "WORK=%TEMP%\basalt-lockpages-%RANDOM%"
mkdir "%WORK%" >nul 2>&1
secedit /export /cfg "%WORK%\current.inf" /areas USER_RIGHTS >nul
if errorlevel 1 (
    echo secedit /export failed.
    rmdir /s /q "%WORK%" >nul 2>&1
    pause
    exit /b 1
)

rem The current holders of the right, e.g. "*S-1-5-32-544,*S-1-5-21-...". type decodes the UTF-16 export.
set "HOLDERS="
for /f "tokens=1,* delims==" %%a in ('type "%WORK%\current.inf" ^| findstr /b /c:"SeLockMemoryPrivilege"') do set "HOLDERS=%%b"
for /f "tokens=* delims= " %%h in ("!HOLDERS!") do set "HOLDERS=%%h"

rem No pipe here: the far side of a pipe runs without delayed expansion. Removing ",*SID," changes the
rem string only when the SID is already a holder (cmd's replacement ignores case).
set "PADDED=,!HOLDERS!,"
if not "!PADDED:,*%SID%,=!"=="!PADDED!" (
    echo "Lock pages in memory" is already granted to %ACCOUNT%.
    goto check
)

if "!HOLDERS!"=="" (set "NEW=*%SID%") else (set "NEW=!HOLDERS!,*%SID%")
(
    echo [Unicode]
    echo Unicode=yes
    echo [Version]
    echo signature="$CHICAGO$"
    echo Revision=1
    echo [Privilege Rights]
    echo SeLockMemoryPrivilege = !NEW!
) > "%WORK%\grant.inf"
secedit /configure /db "%WORK%\grant.sdb" /cfg "%WORK%\grant.inf" /areas USER_RIGHTS /quiet
if errorlevel 1 (
    echo secedit /configure failed; the right was not granted.
    rmdir /s /q "%WORK%" >nul 2>&1
    pause
    exit /b 1
)
echo Granted "Lock pages in memory" to %ACCOUNT%.
echo Sign out and back in ^(or restart^) for it to take effect.

:check
rmdir /s /q "%WORK%" >nul 2>&1
echo.
echo The right in this window's own token (it shows up after the next sign-in):
whoami /priv | findstr /i "SeLockMemoryPrivilege" || echo   not in this token yet
echo.
pause
exit /b 0
