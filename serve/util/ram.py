"""The machine's available RAM, without psutil: GlobalMemoryStatusEx on Windows, MemAvailable elsewhere."""
from __future__ import annotations

import ctypes
import os
import sys


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_bytes() -> int | None:
    """Physical RAM the OS could hand out now, in bytes; None where it gives no reading."""
    if sys.platform == "win32":
        st = _MemoryStatusEx()
        st.dwLength = ctypes.sizeof(st)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        except (AttributeError, OSError):
            pass
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        page, avail = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_AVPHYS_PAGES")
        return int(page * avail)
    except (AttributeError, ValueError, OSError):
        return None
