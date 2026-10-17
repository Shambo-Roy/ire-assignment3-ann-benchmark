"""Diagnostic probe (Windows only, small synthetic data, not a benchmark). Run: .venv/Scripts/python experiments/probes/blas_probe.py
Results recorded in IMPLEMENTATION_LOG.md, design revision B (2026-10-16)."""
import os, sys, time, ctypes, glob, psutil, numpy as np, faiss
p = psutil.Process(); p.cpu_affinity(list(range(12)))
dll = glob.glob(os.path.join(os.path.dirname(faiss.__file__), "..", "faiss_cpu.libs", "libopenblas*.dll"))[0]
ob = ctypes.CDLL(dll)
has = {f: hasattr(ob, f) for f in ("openblas_set_num_threads","openblas_get_num_threads","openblas_get_parallel","openblas_get_config")}
if has["openblas_get_config"]: ob.openblas_get_config.restype = ctypes.c_char_p
print("env OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"), "| exports:", has)
print("faiss openblas threads:", ob.openblas_get_num_threads(), "| parallel mode (0=seq,1=pthreads,2=openmp):", ob.openblas_get_parallel(), "|", ob.openblas_get_config().decode())
rng=np.random.default_rng(0); x=rng.random((60000,64),dtype=np.float32); q=rng.random((2000,64),dtype=np.float32)
def timed(f):
    c0=sum(p.cpu_times()[:2]); t0=time.perf_counter(); f(); w=time.perf_counter()-t0; c=sum(p.cpu_times()[:2])-c0; return w, c/w
def ivf(): qz=faiss.IndexFlatL2(64); i=faiss.IndexIVFPQ(qz,64,256,8,8); i.train(x); i.add(x)
fl=faiss.IndexFlatL2(64); fl.add(x)
def batch(): fl.search(q,10)
def single(): [fl.search(q[j:j+1],10) for j in range(200)]
for omp, blas in ((1,1),(12,1),(1,12),(12,12)):
    faiss.omp_set_num_threads(omp); ob.openblas_set_num_threads(blas)
    r = {n: timed(f) for n,f in (("ivfpq_train_add",ivf),("flat_batch",batch),("flat_single_x200",single))}
    print(f"omp={omp:2d} blas={blas:2d} | " + " | ".join(f"{n} {w:5.2f}s cpu/wall={u:4.1f}" for n,(w,u) in r.items()))
