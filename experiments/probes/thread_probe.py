"""Diagnostic probe (Windows only, small synthetic data, not a benchmark). Run: .venv/Scripts/python experiments/probes/thread_probe.py
Results recorded in IMPLEMENTATION_LOG.md, design revision B (2026-10-16)."""
import ctypes, ctypes.wintypes as wt, time, os, psutil, numpy as np, faiss
k = ctypes.WinDLL("kernel32", use_last_error=True)
class GA(ctypes.Structure): _fields_=[("Mask",ctypes.c_size_t),("Group",wt.WORD),("Reserved",wt.WORD*3)]
def thread_masks():
    out=[]
    for t in psutil.Process().threads():
        h=k.OpenThread(0x0040|0x0200, False, t.id)  # QUERY_INFORMATION|QUERY_LIMITED
        g=GA()
        if h and k.GetThreadGroupAffinity(h, ctypes.byref(g)): out.append(g.Mask)
        k.CloseHandle(h)
    return out
p=psutil.Process()
print("DLLs:", sorted({os.path.basename(m.path).lower() for m in p.memory_maps() if any(s in m.path.lower() for s in ("blas","vcomp","omp","mkl"))}))
P = list(range(12)); p.cpu_affinity(P)
print("process affinity:", p.cpu_affinity())
rng=np.random.default_rng(0); x=rng.random((60000,64),dtype=np.float32); q=rng.random((2000,64),dtype=np.float32)
def hnsw(nt):
    faiss.omp_set_num_threads(nt); i=faiss.IndexHNSWFlat(64,16); t=time.perf_counter(); i.add(x); return time.perf_counter()-t
def ivftrain(nt):
    faiss.omp_set_num_threads(nt); qz=faiss.IndexFlatL2(64); i=faiss.IndexIVFPQ(qz,64,256,8,8); t=time.perf_counter(); i.train(x); i.add(x); return time.perf_counter()-t
def flatbatch(nt):
    faiss.omp_set_num_threads(nt); i=faiss.IndexFlatL2(64); i.add(x); t=time.perf_counter(); i.search(q,10); return time.perf_counter()-t
for name,f in (("HNSW add",hnsw),("IVFPQ train+add",ivftrain),("Flat batch search (BLAS path, nq=2000)",flatbatch)):
    f(1); a=f(1); b=f(12); print(f"{name:42s} 1 thr {a:6.2f}s  12 thr {b:6.2f}s  speedup {a/b:4.1f}x")
masks=thread_masks(); allowed=sum(1<<c for c in P)
print("threads:",len(masks),"| all thread masks within P-core set:", all(m & ~allowed == 0 for m in masks), "| distinct masks:", {hex(m) for m in masks})
faiss.omp_set_num_threads(1); print("omp_get_max_threads after reset:", faiss.omp_get_max_threads())
