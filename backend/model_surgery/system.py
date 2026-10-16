"""Host memory readings from the operating system (no third-party dependencies).

These are measurements, unlike the figures in ``memory.py``. Each function
returns ``None`` when the platform does not expose the value, and callers must
treat that as "unknown" rather than zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel


class SystemMemory(BaseModel):
    total_bytes: int | None
    available_bytes: int | None
    source: str


def system_memory() -> SystemMemory:
    if sys.platform == "win32":
        return _win_system_memory()
    if sys.platform.startswith("linux"):
        return _linux_system_memory()
    return SystemMemory(total_bytes=None, available_bytes=None, source="not available on this platform")


def process_memory(pid: int) -> int | None:
    """Resident memory (working set / VmRSS) of a process, in bytes."""
    try:
        if sys.platform == "win32":
            return _win_process_rss(pid)
        if sys.platform.startswith("linux"):
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _linux_system_memory() -> SystemMemory:
    vals: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                vals[k] = int(parts[0]) * 1024
    except OSError:
        return SystemMemory(total_bytes=None, available_bytes=None, source="/proc/meminfo unreadable")
    return SystemMemory(total_bytes=vals.get("MemTotal"), available_bytes=vals.get("MemAvailable"), source="/proc/meminfo")


def _win_system_memory() -> SystemMemory:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return SystemMemory(total_bytes=None, available_bytes=None, source="GlobalMemoryStatusEx failed")
    return SystemMemory(total_bytes=st.ullTotalPhys, available_bytes=st.ullAvailPhys, source="GlobalMemoryStatusEx")


def _win_process_rss(pid: int) -> int | None:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_VM_READ = 0x0010
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.OpenProcess.restype = wintypes.HANDLE
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        c = PROCESS_MEMORY_COUNTERS()
        c.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return None
        return int(c.WorkingSetSize)
    finally:
        kernel32.CloseHandle(h)
