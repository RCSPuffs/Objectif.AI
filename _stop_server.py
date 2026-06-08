"""
Helper called by remove-service.bat.
Finds and terminates python.exe running main.py from this app's directory.
Usage: python _stop_server.py <app_dir>
"""
import sys
import os
import subprocess
import time

appdir = sys.argv[1].rstrip('\\').rstrip('/')
main_py = os.path.join(appdir, 'main.py')

print(f"  Looking for python.exe running: {main_py}")

killed = 0
try:
    # Get all python.exe processes with their PIDs and command lines
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'",
         'get', 'ProcessId,CommandLine', '/format:csv'],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.splitlines():
        # CSV format: Node,CommandLine,ProcessId
        if 'main.py' in line and appdir.lower() in line.lower():
            parts = line.strip().split(',')
            if len(parts) >= 3:
                pid = parts[-1].strip()
                if pid.isdigit():
                    print(f"  Killing PID {pid}")
                    subprocess.run(['taskkill', '/F', '/PID', pid],
                                   capture_output=True)
                    killed += 1
except Exception as e:
    print(f"  wmic approach failed ({e}), trying psutil...")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    cwd = proc.info['cwd'] or ''
                    if appdir.lower() in cmdline.lower() or appdir.lower() in cwd.lower():
                        print(f"  Killing PID {proc.info['pid']}")
                        proc.kill()
                        killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        print("  psutil not available either — process may still be running")

# Also kill wscript.exe running our VBS
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='wscript.exe'",
         'get', 'ProcessId,CommandLine', '/format:csv'],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.splitlines():
        if 'start-silent.vbs' in line and appdir.lower() in line.lower():
            parts = line.strip().split(',')
            if len(parts) >= 3:
                pid = parts[-1].strip()
                if pid.isdigit():
                    subprocess.run(['taskkill', '/F', '/PID', pid],
                                   capture_output=True)
except Exception:
    pass

if killed == 0:
    print("  No running Objectif.AI process found (already stopped?)")
else:
    print(f"  Stopped {killed} process(es)")
    time.sleep(2)
