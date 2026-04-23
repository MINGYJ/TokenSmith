import ctypes

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullExtendedVirtual', ctypes.c_ulonglong),
    ]

m = MEMORYSTATUSEX()
m.dwLength = ctypes.sizeof(m)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print(f"RAM:          {m.ullTotalPhys/1e9:.1f} GB total,  {m.ullAvailPhys/1e9:.1f} GB avail")
print(f"Page file:    {m.ullTotalPageFile/1e9:.1f} GB total,  {m.ullAvailPageFile/1e9:.1f} GB avail")
print(f"Virtual space:{m.ullTotalVirtual/1e9:.1f} GB total,  {m.ullAvailVirtual/1e9:.1f} GB avail")

# Check page file setting
import subprocess
result = subprocess.run(
    ["wmic", "pagefile", "list", "full"],
    capture_output=True, text=True
)
print("\nPage file config:")
print(result.stdout)
