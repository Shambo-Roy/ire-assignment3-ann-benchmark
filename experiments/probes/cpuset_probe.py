"""Diagnostic probe (Windows only, small synthetic data, not a benchmark). Run: .venv/Scripts/python experiments/probes/cpuset_probe.py
Results recorded in IMPLEMENTATION_LOG.md, design revision B (2026-10-16)."""
import ctypes, struct, psutil
k = ctypes.WinDLL("kernel32", use_last_error=True)
n = ctypes.c_ulong(0)
k.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
buf = ctypes.create_string_buffer(n.value)
assert k.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0)
off, rows = 0, []
while off < n.value:
    size, typ = struct.unpack_from("<II", buf, off)
    if typ == 0:
        cid, grp, lp, core, llc, numa, eff = struct.unpack_from("<IHBBBBB", buf, off + 8)
        rows.append((lp, core, eff))
    off += size
for lp, core, eff in rows: print(f"logical={lp:2d} core={core:2d} efficiency_class={eff}")
from collections import Counter
print("cores per efficiency class:", Counter(e for _, c, e in {(0, c, e) for _, c, e in rows}))
p = psutil.Process(); print("current affinity:", p.cpu_affinity())
p.cpu_affinity([rows[0][0]]); print("set ->", p.cpu_affinity())
