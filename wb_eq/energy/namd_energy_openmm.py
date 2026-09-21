#!/usr/bin/env python3

# ========================================================================
# OpenMM and MDAnalysis implementation of NAMD PairInteraction Energy
# ------------------------------------------------------------------------
# OPTIMIZED FOR GPU's
# ----------------------
# => calculate pairinteraction energies from NAMD simulation trajectories with CHARMM force fields
# => STATIC and DYNAMIC selections, self and cross-interactions
# => PME for long range electrostatics (NAMD pairinteraction does not have this)
#    Electrostatic energies with PME will be highly negative compared to NAMD pairinteraction
# => Trajectory Disk Streaming/RAM loading/RAM chunking modes (chunking requires catdcd)
# => Smart multithreading allocation
# ------------------------------------------------------------------------

## USAGE --------------------------------------------------
# 0: First run normal simulation to obtain .dcd trajectories
# 1. Copy script to working dir
# 2. INPUT: Set input structure (.psf) and trajectories (.dcd)
# 3. INPUT: Set selection 1, Selection 2 (Optional), out_energies (Optional), out_file_prefix
#----------------------------------------------------------------------------
# -> ALTERNATIVELY, SET ENVIRONMENT VARIABLES (used when variables are not set in script)
#----------------------------------------------------------------------------
#	-> NAMD_ENERGY_SELECTION1	        =	selection1
#	-> NAMD_ENERGY_SELECTION2	        =	selection2 		  (optional)
#	-> NAMD_ENERGY_UPDATE_SELECTION1	=	update_selection1
#	-> NAMD_ENERGY_UPDATE_SELECTION2	=	update_selection2
#	-> NAMD_ENERGY_OUT_ENERGIES	        = 	out_energies	  (optional)
#	-> NAMD_ENERGY_OUT_PREFIX	        =	out_file_prefix
#	-> NAMD_ENERGY_TIMESTEP_END         =	timestep_end      (optional)
#	-> NAMD_ENERGY_LABEL                =	label			  (optional)
#   -> NAMD_ENERGY_PROCESSES            =   namd_processes    (optional)
#----------------------------------------------------------------------------
# 4. set other input and output params [search for TODO]
# 5. run with "./namd_energy_openmm.py"
# 	OR
# 6. use namd_energy_openmm.sh launcher.
#	-> First, unset selection1, selection2, out_energies, out_file_prefix in this script
#	-> Set environment variables in namd_energy_openmm.sh
#		=> ./namd_energy_openmm.sh

import os
import sys

# Helper function to find dcd files in a folder. min_num and max_num are both inclusive
def find_files(dir_path, prefix, suffix, min_num=None, max_num=None, sort_natural=True, return_abs_path=False):
    import re; from pathlib import Path
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir(): raise FileNotFoundError(f"Directory '{dir_path}' not found.")
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    result_list = []
    for f in dir_path_obj.iterdir():
        if f.is_file():
            match = pattern.search(f.name)
            if match:
                num = int(match.group(1))
                if (min_num is None or num >= min_num) and (max_num is None or num <= max_num):
                    result_list.append(str(f.resolve()) if return_abs_path else str(f.relative_to(".")))

    if sort_natural:
        def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        result_list.sort(key=natural_sort_key)
    return result_list


DEBUG: bool = True
# --------------------------------------------------------------------
# PERFORMANCE
# --------------------------------------------------------------------
USE_GPU: bool = True                 # Keep it True, Auto-fallbacks to CPU
NUM_COMPUTE_WORKERS: int = 4         # [GPU ONLY] TODO: OpenMM Contexts running in parallel (Consume RAM and VRAM). Scale up for DYNAMIC MODE

RAM_LOAD_ENABLED: bool = True        # TODO: Loads DCD files to RAM_DISK before processing, bypasses I/O bottlenecks
RAM_CHUNK_MODE: bool = True          # Loads big DCD files to RAM_DISK in chunks. REQUIRES CATDCD in PATH
RAM_READER_COUNT: int = 2            # Concurrent readers for RAM chunks. Auto-decrement for small trajectories (chunks)


# --------------------------------------------------------------------
# INPUT
# --------------------------------------------------------------------
PARAM_FILES = [
    "../../common/ff/par_all36m_prot.prm",
    "../../common/ff/toppar_water_ions.prot.str"
]
PSF_FILE = "../../common/amyl_wb.psf"       # TODO : input structure file
DCD_FILES = find_files("..", "amyl_wb_eq", ".dcd")          # TODO : trajectory dcd files

## Selections (MDAnalysis selection syntax)
# -> hydration shell: water and around 4.25 protein
SELECTION1: str = os.getenv("NAMD_ENERGY_SELECTION1", "")   # TODO: or set ENV VAR: NAMD_ENERGY_SELECTION1
SELECTION2: str = os.getenv("NAMD_ENERGY_SELECTION2", "")   # TODO: or set ENV VAR: NAMD_ENERGY_SELECTION2

# Dynamic Selections (update every frame)
UPDATE_SELECTION1: str = str(os.getenv("NAMD_ENERGY_UPDATE_SELECTION1", 0))     # TODO
UPDATE_SELECTION2: str = str(os.getenv("NAMD_ENERGY_UPDATE_SELECTION2", 0))     # TODO
FAST_DYNAMIC_SELECTION: bool = True   # Bypasses slow MDAnalysis 'updating=True' and manually update selection using FastDynamicSelector

## Output file names
OUT_FILE_PREFIX: str = os.getenv("NAMD_ENERGY_OUT_PREFIX", "interaction")    # TODO: or set ENV VAR: NAMD_ENERGY_OUT_PREFIX

## [OPTIONAL][ Label for this run
LABEL: str = os.getenv("NAMD_ENERGY_LABEL", "Interaction Energy (OpenMM)")   # or ser ENV VAR: NAMD_ENERGY_LABEL

### Energies to calculate (as sequence of 4-letter codes)
# -------------------------------------------------------------------------------------
# OPTIONS: -bond -angl -dihe -impr -conf -vdw -elec -nonb -pote -all
# -------------------------------------------------------------------------------------
# -> -dihe (dihedral), -impr (imporper), -pote (potential)
# -> -conf (confomational) = bond + angle + dihedral + improper
# -> -nonb (non-nonded)    = elec + vdw
# -> -pote (potential)     = conf + nonb
# -------------------------------------------------------------------------------------
# WITH SELECTION 2: ONLY [ -vdw -elec -nonb -pote -all ] ARE ALLOWED

OUT_ENERGIES: list[str] = os.getenv("NAMD_ENERGY_OUT_ENERGIES", "-all").split()     # TODO: or set ENV VAR: NAMD_ENERGY_OUT_ENERGIES

## Params
CUTOFF: float = 12.0            # TODO: Cutoff distance (in Å)
SWITCHDIST: float = 10.0        # TODO: Switch distance (in Å) for non-bonded interactions. 0 to trn off switching
DIELECTRIC: float = 1.0         # ielectric constant (> 1 will lessen the electrostatic forces)
TEMPERATURE: float = 300        # [Optional] Temperature (in K) (Only used for bookkeeping)

## Periodic [OPTIONAL]
PERIODIC: bool = True
PME_ENABLED: bool = True        # [ONLY PERIODIC] PME for long-range electrostatics

## TIme Step parameters (ONLY USED FOR OUTPUT COLUMNS, DOES NOT AFFECT CALCULATION)
TIMESTEP_FIRST: int = 0         # only for bookkeeping
FRAME_FREQ: int = 100           # TODO: timesteps between frames (=dcd_freq). only for bookkeeping

# Skip Frames, faster calculation
FRAME_SKIP: int = 0             # TODO: frame_step = frame_skip + 1

# --------------------------------------------------------------------
# OUTPUT Params
# --------------------------------------------------------------------
## Force Output	(ONLY APPLICABLE when SEL-2 is defined)
# -> Calculates force on SEL-1 due to SEL-2
OUT_FORCE: bool = True
OUT_FORCE_COMPONENTS: bool = False       # output force XYZ components

# total force will be the VECTOR SUM: mag(total_force) = mag(vdw_force + elec_force vectors) [true physical behaviour].
# Else, mag(total_force) = mag(vdw_force) + mag(elec_force)    [NO CANCELLATIONS, PHYSICALLY INACCURATE]
TOTAL_FORCE_VECTOR_SUM: bool = True

OUT_ENERGY_FORMAT = "{:.4f}"
OUT_DELIMITER = " "
COMMENT_TOKEN = "#"

# --------------------------------------------------------------------
# FRAME LOADING and PERFORMANCE
# --------------------------------------------------------------------
RAM_DISK_PATH = "/tmp/namd_energy.openmm"          # ram disk path to use

QUEUE_FRAME_COUNT = 50                # Size of the Zero-Copy Shared Memory Ring Buffer (Reduce if using 1M+ atoms)

## RAM CHUNK MODE (requires catdcd)
RAM_CHUNK_DYNAMIC: bool = True        # Automatically shrink chunk if RAM is constrained
RAM_CHUNK_FRAMES: int = 10000         # Max frames per chunk
RAM_CHUNK_MIN_FRAMES: int = 2000      # Fallback to disk streaming if chunks cannot meet this size

RAM_SAFETY_MARGIN_GB = 1.0            # Base free RAM margin required (GiB)
RAM_EXTRA_MARGIN_GB = 0.1             # Extra buffer headroom (GiB)

## Thread controls
# 0 = Smart Auto-Allocation, >0 = Override
MDA_THREADS_PER_READER = 0            # Threads per MDAnalysis reader instance (OpenMP)
OPENMM_THREADS_PER_CONTEXT = 0        # Compute threads for each OpenMM context (applies only if running on CPU)


# -----------------------------------
# EXtra Options
# -----------------------------------
RAM_READER_MIN_FRAMES = 50     # Minimum frames a RAM reader must have to read, otherwise dynamically lower reader count

PROGRESS_REPORT_INTERVAL_FRAMES: int = 1000      # num frames

INDEX_STREAM_BUFFER_CHUNK_SIZE: int = 2000       # num of frames to hold the computed energy data in RAM
INDEX_STREAM_BUFFER_ALWAYS_OPEN: bool = True     # keep output file open

# Optimize Memory: Force masking for large interaction pair count to avoid OOM crashes
MAX_INTERACTION_PAIRS_IN_RAM = 250_000_000      # consumes 4-8 bytes per interaction pair
INTERACTION_PAIR_MEMORY_MB = 8.0 / (1024 ** 2)  # memory (MiB) per interaction pair

MANUAL_GC_ENABLED: bool = True                  # Manual GC (DYNAMIC MODE ONLY)
MANUAL_GC_INTERVAL_FRAMES: int = 5000           # num frames

## Experimental Features -----------
## experimental flag to tun off erfc(ewald_beta * r) factor in short range direct electrostatics
# if true: multiplies short range raw coulomb energy with erfc(ewald_beta * r) (very fast decaying factor)
# else: uses raw coulomb energy expression for short range electrostatics with sharp discontinuity at the cutoff
PME_SHORT_RANGE_USE_EWALD_BETA: bool = True
PME_TOLERANCE: float = 1e-6                      # NAMD default PME error tolerance (unitless factor)




# ==========================================================================
# MAIN
# ==========================================================================
print("")

## Logging ------------------------------
NOCOL = "\033[0m"   # reset color
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

def log_info(msg): print(f"{GREEN}[INFO]{NOCOL} {msg}")

def log_debug(msg):
    if DEBUG: print(f"{CYAN}[DEBUG]{NOCOL} {msg}")


def log_warn(msg, exc=None):
    if exc is not None:
        import traceback
        traceback.print_exception(exc)
    print(f"{YELLOW}[WARN]{NOCOL} {msg}")


def log_error(msg, exc=None):
    if exc is not None:
        import traceback
        traceback.print_exception(exc)
    print(f"{RED}[ERROR]{NOCOL} {msg}")
    sys.exit(1)


# --------------------------------------------
# HARDWARE INIT and CHECKS
# --------------------------------------------
# OpenMM Hardware Initialization (CUDA -> HIP -> OpenCL -> CPU)
from openmm import Platform

OPENMM_PLATFORM_NAME = "CPU"
OPENMM_PLATFORM_DISPLAY_NAME = "CPU"
OPENMM_PLATFORM_PROPERTIES = {}
if USE_GPU:
    _platform = None
    for gpu_plat_name in ['CUDA', 'HIP', 'OpenCL']:
        try:
            _platform = Platform.getPlatformByName(gpu_plat_name)
            OPENMM_PLATFORM_NAME = gpu_plat_name
            OPENMM_PLATFORM_DISPLAY_NAME = f"{gpu_plat_name} (GPU)"
            break
        except Exception:
            continue
    if _platform is None:
        USE_GPU = False
        log_warn("GPU requested but CUDA, HIP, and OpenCL are unavailable. Falling back to CPU.")
    del _platform        # GC
    # gc.collect()

log_info(f"Auto-detected OPENMM PLATFORM: {OPENMM_PLATFORM_DISPLAY_NAME}  |  USE_GPU={USE_GPU}")

if NUM_COMPUTE_WORKERS < 1:
    log_warn("Number of OpenMM contexts (compute workers) must be >= 1. Resetting to 1 context")
    NUM_COMPUTE_WORKERS = 1
elif NUM_COMPUTE_WORKERS > 1 and not USE_GPU:
    log_warn("Multiple OpenMM contexts (compute workers) on CPU will severally stall. Resetting to 1 context")
    NUM_COMPUTE_WORKERS = 1

if USE_GPU and NUM_COMPUTE_WORKERS == 1:
    log_warn("STALL_WARNING: GPU may stall with only 1 OpenMM context (compute worker). You may want to increase NUM_COMPUTE_WORKERS")


# Thread Allocation ----------------------------------
# Must be before all major imports
SYS_CORES = os.cpu_count() or 4
actual_ram_reader_count = max(min(RAM_READER_COUNT, SYS_CORES - 1), 1) if RAM_LOAD_ENABLED else 1
if actual_ram_reader_count != RAM_READER_COUNT:
    log_warn(f"RAM READER COUNT: Changed from {RAM_READER_COUNT} => {actual_ram_reader_count} due to system limits")

# OpenMM Thread allocator (CPU only)
openmm_alloc_mode = "Smart Auto"
if USE_GPU:
    assigned_openmm_threads = 1
    if OPENMM_THREADS_PER_CONTEXT > 1:
        log_info(f"Running OpenMM on GPU, IGNORING OPENMM_THREADS_PER_CONTEXT={OPENMM_THREADS_PER_CONTEXT}")
else:
    if OPENMM_THREADS_PER_CONTEXT > 0:
        assigned_openmm_threads = OPENMM_THREADS_PER_CONTEXT
        openmm_alloc_mode = "User Override"
        if (assigned_openmm_threads * NUM_COMPUTE_WORKERS) > SYS_CORES:
            log_warn(f"STALL WARNING: Running {NUM_COMPUTE_WORKERS} OpenMM CPU context (compute workers) with {assigned_openmm_threads} threads/context, but system only has {SYS_CORES} threads")
    else:
        assigned_openmm_threads = max(1, int(SYS_CORES * 0.80 / NUM_COMPUTE_WORKERS))

# MDA Thread Allocator
if MDA_THREADS_PER_READER > 0:
    assigned_mda_threads = MDA_THREADS_PER_READER
    mda_alloc_mode = "User Override"
else:
    remaining_cores = max(1, SYS_CORES - (assigned_openmm_threads * NUM_COMPUTE_WORKERS))
    assigned_mda_threads = max(1, remaining_cores // actual_ram_reader_count)
    mda_alloc_mode = "Smart Auto"

## FINALIZE THREAD ALLOCATION --------------
# set OpenMM CPU mode Threads
if OPENMM_PLATFORM_NAME == "CPU":
    OPENMM_PLATFORM_PROPERTIES['Threads'] = str(assigned_openmm_threads)

# Set OpenMP thread limit prior to loading C-extensions of MDAnalysis
os.environ["OMP_NUM_THREADS"] = str(assigned_mda_threads)


# Logging
if __name__ == '__main__':
    print("-" * 60)
    log_info(f"OPENMM CONFIGURATION (Compute Engine)")
    log_info(f" => Platform    : {OPENMM_PLATFORM_DISPLAY_NAME}")
    log_info(f" => Contexts    : {NUM_COMPUTE_WORKERS} (compute workers)")
    log_info(f" => CPU Threads : {assigned_openmm_threads}/context  ({openmm_alloc_mode})")
    print("-" * 60)
    log_info(f"MDAnalysis CONFIGURATION (Frame Reader)")
    log_info(f" => RAM Loading : {f'ON  (Chunking: {RAM_CHUNK_MODE})' if RAM_LOAD_ENABLED else 'OFF'}")
    log_info(f" => Readers     : {f'{actual_ram_reader_count} (RAM cached mode) |' if RAM_LOAD_ENABLED else ''} 1 (DISK stream)")
    log_info(f" => CPU Threads : {assigned_mda_threads}/reader ({mda_alloc_mode})  [OMP_NUM_THREADS]")
    print("-" * 60)
# ---------------------------------------------------------------------

# Imports (must be after thread allocation)
import gc
import time
import queue
import math
import re
import uuid
import shutil
import signal
import atexit
import threading

from itertools import chain
import subprocess
import numpy as np
import openmm as mm
from openmm import app, unit

import warnings
warnings.filterwarnings("ignore", message=r".*DCDReader currently makes independent timesteps.*")
import MDAnalysis as mda

import multiprocessing as mp
try:
    # Force 'fork' to prevent catastrophic script re-execution on macOS/Windows spawn defaults
    mp.set_start_method('fork', force=True)
except Exception:
    pass
from multiprocessing import shared_memory



# Helper functions and classes ------------------------------
def boolify(value: str, default_val: bool = False, err_msg: str = "") -> bool:
    value = value.strip().lower()
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    if value in truthy: return True
    if value in falsy: return False
    if err_msg.strip():
        log_error(f"{err_msg}: {value!r}")
        raise ValueError(f"{err_msg}: {value!r}")
    return default_val


def get_cur_datetime_formatted() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =======================================================
# FAST DYNAMIC SELECTOR (C-Level Distance Engine)
# =======================================================
class FastDynamicSelector:

    def __init__(self, selection_string: str, universe: mda.Universe):
        self.selection_string = selection_string
        self._triclinic_warned = False

        # Parse the string: optional base sel + and/or + optional not + around + dist + ref sel
        pattern = re.compile(r"^(?:(.*?)\s+(?:and|or)\s+)?(not\s+)?around\s+([0-9.]+)\s+(.*)$", re.IGNORECASE)
        match = pattern.match(selection_string.strip())

        if not match:
            raise ValueError(
                f"Could not parse dynamic selection: '{selection_string}'. Expected format: '<base_sel> and [not] around <distance> <ref_sel>'")

        base_str, not_str, dist_str, ref_str = match.groups()

        self.base_sel_str = base_str.strip() if base_str else "all"
        self.invert = bool(not_str and "not" in not_str.lower())
        self.cutoff = float(dist_str)
        self.ref_sel_str = ref_str.strip()

        if not self.ref_sel_str:
            raise ValueError("Reference selection missing in 'around' clause.")

        log_debug(
            f"FastDynamicSelector parsed: Base='{self.base_sel_str}', Invert={self.invert}, Cutoff={self.cutoff}A, Ref='{self.ref_sel_str}'")

        # Extract permanent indices at frame 0
        base_ag = universe.select_atoms(self.base_sel_str)
        ref_ag = universe.select_atoms(self.ref_sel_str)

        self.base_indices = base_ag.indices.copy()
        self.ref_indices = ref_ag.indices.copy()

        if len(self.base_indices) == 0: log_warn(f"Base selection '{self.base_sel_str}' yielded 0 atoms at frame 0.")
        if len(self.ref_indices) == 0: log_warn(f"Reference selection '{self.ref_sel_str}' yielded 0 atoms at frame 0.")

    def eval(self, u: mda.Universe, is_periodic: bool) -> np.ndarray:
        if len(self.base_indices) == 0 or len(self.ref_indices) == 0:
            return np.array([], dtype=np.int64) if not self.invert else self.base_indices

        # Fetch raw C-level coordinates instantly
        coords = u.trajectory.ts.positions
        dimensions = u.trajectory.ts.dimensions if is_periodic else None    # 1D array of len 6 [lx, ly, lz, alpha, beta, gamma]

        pos_base = coords[self.base_indices]
        pos_ref = coords[self.ref_indices]

        use_scipy = True
        boxsize = None
        if dimensions is not None:
            # SciPy cKDTree only supports orthogonal periodic boxes (angles == 90)
            if np.allclose(dimensions[3:], 90.0):
                boxsize = dimensions[:3]
            else:
                use_scipy = False
                if not self._triclinic_warned:
                    log_warn(
                        "Triclinic box detected! SciPy cKDTree requires orthogonal boxes. Falling back to MDAnalysis capped_distance (Slower).")
                    self._triclinic_warned = True

        if use_scipy:
            # SciPy strictly requires coordinates to be inside the [0, boxsize) domain.
            # We must wrap them using modulo arithmetic.
            if boxsize is not None:
                pos_ref = pos_ref % boxsize
                pos_base = pos_base % boxsize

                # IEEE 754 Float Quirk: Modulo can occasionally yield a value exactly equal to boxsize.
                # In periodic bounds, boxsize is physically identical to 0.0, so we snap it back.
                pos_ref[pos_ref >= boxsize] = 0.0
                pos_base[pos_base >= boxsize] = 0.0

            # 1. Build C++ spatial grid on the reference (Protein)
            from scipy.spatial import cKDTree
            tree = cKDTree(pos_ref, boxsize=boxsize)

            # 2. Query with base (Water).
            # k=1 (only find the single nearest atom).
            # distance_upper_bound early-exits math if further than cutoff.
            # workers=-1 automatically multithreads across all CPU cores.
            dists, _ = tree.query(pos_base, k=1, distance_upper_bound=self.cutoff, workers=1)

            # 3. Create boolean mask (values > cutoff are returned as 'inf' by SciPy)
            valid_mask = dists <= self.cutoff
            selected_actual_idx = self.base_indices[valid_mask]
        else:
            # Fallback for Triclinic/Non-Orthogonal boxes
            from MDAnalysis.lib.distances import capped_distance
            pairs = capped_distance(pos_base, pos_ref, self.cutoff, box=dimensions, return_distances=False)
            if len(pairs) > 0:
                unique_base_idx = np.unique(pairs[:, 0])
                selected_actual_idx = self.base_indices[unique_base_idx]
            else:
                selected_actual_idx = np.array([], dtype=np.int64)

        if self.invert:
            selected_actual_idx = np.setdiff1d(self.base_indices, selected_actual_idx)

        return selected_actual_idx


# =======================================================
# ZERO-COPY SHARED MEMORY RING BUFFER (IPC Architecture)
# =======================================================
class SharedFrameBuffer:
    """
    Zero-Copy Inter-Process Communication (IPC) ring buffer.
    Allocates contiguous RAM blocks to share coordinate and index arrays
    across independent Python processes without Pickling/Serialization overhead.

    WORKFLOW:
    ----------------------
    => [Producer process]
    ----------------------
    slot_idx = None
    while not shutdown_event.is_set():
        try:
            # Lease a free memory block from the ring buffer
            slot_idx = shm_buffer.free_slots.get(timeout=1.0)
            break
        except queue.Empty: continue

    if shutdown_event.is_set() or slot_idx is None: break OR return

    # Dump data directly into raw RAM
    shm_buffer.write_frame(slot_idx, abs_f, coords_nm, box_nm, arr1, arr2)

    # Notify Compute Workers that this memory block is ready
    shm_buffer.ready_slots.put(slot_idx)

    ----------------------
    => [Consumer process]
     ----------------------
    while not shutdown_event.is_set():
        try:
            slot_idx = shm_buffer.ready_slots.get(timeout=1.0)
        except queue.Empty: continue

        if slot_idx is None: break

        # Zero-Copy Read from Shared RAM. DO NOT FREE IT YET, OTHERWISE it may get overwritten by producer
        abs_f, coords_nm, box_nm, arr_sel1, arr_sel2, n1, n2 = shm_buffer.read_frame(slot_idx)

        # DO WORK WITH THE SHARED DATA HERE.

        # When done., release memory
        shm_buffer.free_slots.put(slot_idx)
    -----------------------------------------------------------------------------

    Must call .close() when done 
    """
    TAG = "SharedFrameBuffer"

    def __init__(self, num_slots: int, n_atoms: int, has_dyn_sel1: bool, has_dyn_sel2: bool, is_creator: bool = False):
        self.num_slots = num_slots
        self.n_atoms = n_atoms
        self.has_dyn_sel1 = has_dyn_sel1
        self.has_dyn_sel2 = has_dyn_sel2
        self.is_creator = is_creator

        self._lock = mp.Lock()
        self._is_closed: bool = False
        self.shm_blocks: dict = {}
        self.arrays: dict = {}

        # Memory Optimization: Use float32 for coords and int32 for indices.
        self.specs = {
            'coords': ((num_slots, n_atoms, 3), np.float32),
            'box': ((num_slots, 3, 3), np.float32),
            'meta': ((num_slots, 3), np.int64)  # Format: [abs_f, n1, n2]
        }

        # Memory Optimization: Only allocate arrays for selections that dynamically change
        if self.has_dyn_sel1:
            self.specs['sel1'] = ((num_slots, n_atoms), np.int32)
        if self.has_dyn_sel2:
            self.specs['sel2'] = ((num_slots, n_atoms), np.int32)

        for name, (shape, dtype) in self.specs.items():
            nbytes = math.prod(shape) * np.dtype(dtype).itemsize
            shm_name = f"namd_energy_shm_{name}"

            if self.is_creator:
                try:
                    # Clean up dangling shared memory from previous crashed runs
                    shared_memory.SharedMemory(name=shm_name).unlink()
                except FileNotFoundError:
                    pass

                log_debug(f"{self.TAG}: Allocating Shared RAM '{shm_name}' ({nbytes / (1024 ** 2):.2f} MB)")
                shm = shared_memory.SharedMemory(create=True, name=shm_name, size=nbytes)
            else:
                shm = shared_memory.SharedMemory(name=shm_name)

            self.shm_blocks[name] = shm
            # Bind a NumPy view directly to the physical RAM block
            self.arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        if self.is_creator:
            # Token rings for safe concurrent access
            self.free_slots = mp.Queue(maxsize=num_slots)
            self.ready_slots = mp.Queue(maxsize=num_slots)
            for i in range(num_slots):
                self.free_slots.put(i)

    def is_closed(self) -> bool: return self._is_closed

    def capacity(self): return self.num_slots

    def size(self): return self.ready_slots.qsize()

    def __len__(self): return self.size()

    def is_empty(self): return self.size() == 0

    def is_full(self): return self.size() == self.capacity()

    def check_creator_closed(self, func_tag=""):
        if not self.is_creator: raise RuntimeError(f"{self.TAG}:{func_tag} Not a creator")
        if self._is_closed: raise RuntimeError(f"{self.TAG}:{func_tag} Already closed")

    def get_free_slot(self, block: bool = True, timeout: float | None = None) -> int:
        self.check_creator_closed("get_free_slot")
        return self.free_slots.get(block=block, timeout=timeout)

    def get_ready_slot(self, block: bool = True, timeout: float | None = None) -> int:
        self.check_creator_closed("get_ready_slot")
        return self.ready_slots.get(block=block, timeout=timeout)

    def put_free_slot(self, slot_idx: int, block: bool = True, timeout: float | None = None):
        self.check_creator_closed("put_free_slot")
        return self.free_slots.put(slot_idx, block=block, timeout=timeout)

    def put_ready_slot(self, slot_idx: int, block: bool = True, timeout: float | None = None):
        self.check_creator_closed("put_ready_slot")
        return self.ready_slots.put(slot_idx, block=block, timeout=timeout)

    def write_frame(self, free_slot_idx: int, abs_frame_idx: int, coords: np.ndarray, box: np.ndarray, arr_sel1: np.ndarray,
                    arr_sel2: np.ndarray):
        """
        Called by Reader Processes: Dumps extracted data directly into the shared RAM slot.

        @:param free_slot_idx: slot id, previously acquired from polling self.free_slots
        @:param abs_f: absolute frame index
        @:param coords_nm: coordinates array. Shape (N_atoms, 3)
        @:param box_nm: box vectors. A 3x3 Matrix
        @:param arr_sel1: selection-1 atom indices array, or NOne
        @:param arr_sel2: selection-2 atom indices array, or None
        """
        # if self._is_closed: raise RuntimeError(f"{self.TAG}: Already closed. Cannot write frames")
            
        n1 = len(arr_sel1) if arr_sel1 is not None else 0
        n2 = len(arr_sel2) if arr_sel2 is not None else 0

        self.arrays['meta'][free_slot_idx, 0] = abs_frame_idx
        self.arrays['meta'][free_slot_idx, 1] = n1
        self.arrays['meta'][free_slot_idx, 2] = n2

        # Implicitly casts float64 coordinates to float32 natively
        self.arrays['coords'][free_slot_idx] = coords

        if box is not None:
            self.arrays['box'][free_slot_idx] = box

        if self.has_dyn_sel1 and n1 > 0:
            self.arrays['sel1'][free_slot_idx, :n1] = arr_sel1
        if self.has_dyn_sel2 and n2 > 0:
            self.arrays['sel2'][free_slot_idx, :n2] = arr_sel2

    def read_frame(self, ready_slot_idx: int):
        """
        Called by Compute Workers: Returns NumPy views sliced exactly to the dynamic selection lengths

        @:param free_slot_idx: slot id previously acquired from polling self.ready_slots
        """
        # if self._is_closed: raise RuntimeError(f"{self.TAG}: Already closed. Cannot read frames")
        
        abs_f = int(self.arrays['meta'][ready_slot_idx, 0])
        n1 = int(self.arrays['meta'][ready_slot_idx, 1])
        n2 = int(self.arrays['meta'][ready_slot_idx, 2])

        coords_nm = self.arrays['coords'][ready_slot_idx]
        box_nm = self.arrays['box'][ready_slot_idx]

        arr_sel1 = self.arrays['sel1'][ready_slot_idx, :n1] if self.has_dyn_sel1 else None
        arr_sel2 = self.arrays['sel2'][ready_slot_idx, :n2] if self.has_dyn_sel2 else None

        return abs_f, coords_nm, box_nm, arr_sel1, arr_sel2, n1, n2

    def close(self):
        if self._is_closed: return
        
        with self._lock:
            if self._is_closed: return
            for name, shm in self.shm_blocks.items():
                shm.close()
                if self.is_creator:
                    try:
                        shm.unlink()
                        log_debug(f"{self.TAG}: Unlinked Shared RAM '{shm.name}'")
                    except Exception as e:
                        log_warn(f"{self.TAG}: Failed to unlink Shared RAM '{shm.name}': {e}", e)

            self._is_closed = True


# =======================================================
# IndexStreamBuffer
# =======================================================
class IndexStreamBuffer:
    """
    A data structure to write the values in order of indices

    Data can come in any order asynchronously, and the job of this class is to look of
    consecutive chunk of indices, and if found, write them to the output file and release memory

    It is thread-safe, but not process safe (for that, replace threading.Lock with mp.Lock, will be slow)
    .close() must be called to flush remaining indices safely.

    -> It stores mappings of index -> value in a dict
    -> checks if a contiguous chunk of consecutive indices exists
    -> dumps the block to output file, and free up memory

    See .insert(index: int, value: object)
        .close()
    """

    TAG = "IndexStreamBuffer"

    def __init__(self,
                 output_file_path: str,
                 chunk_size: int,
                 keep_file_open: bool = False,
                 string_converter_callback=None,
                 pre_chunk_write_callback=None,
                 post_chunk_write_callback=None):

        """
        :parameter output_file_path: output file path
        :parameter chunk_size: a contiguous block of indices to write to output file
        :parameter string_converter_callback: a function that converts actual data to strings for writing
        :parameter pre_chunk_write_callback: a function that is called before a chunk is written to output file.
                   It takes chunk_index as input, and may return a string  to be written before the chunk
        :parameter post_chunk_write_callback: a function that is called after a chunk is written to output file.
                   It takes chunk_index and chunk_size (number of entries in the chunk)_as input, and returns nothing
        :parameter keep_file_open:  if true, file descriptor is kept open for the entire time until close() is called
        """

        if output_file_path is None or len(output_file_path.strip()) == 0:
            raise ValueError(f"{self.TAG}: output_file_path cannot be empty")
        if chunk_size <= 0:
            raise ValueError(f"{self.TAG}: Chunk size must be greater than zero. Given: {chunk_size}")

        self.out_file_path = output_file_path
        self.chunk_size = chunk_size
        self.string_converter_callback = string_converter_callback if string_converter_callback is not None else str
        self.pre_chunk_write_callback = pre_chunk_write_callback
        self.post_chunk_write_callback = post_chunk_write_callback
        self.keep_file_open = keep_file_open

        # internal state
        self._written_chunk_count: int = 0
        self._out_fd = None  # only when keep_file_open is true
        self._is_closed: bool = False
        self._lock = threading.Lock()

        self._data: dict = {}
        self._next_index: int = 0
        # self._next_range_set: set = set()
        self._set_next_index(0)  # init next index

    def _consider_delete_file(self):
        if self._written_chunk_count == 0 and os.path.exists(self.out_file_path):
            try:
                os.remove(self.out_file_path)
            except Exception as e:
                raise RuntimeError("{self.TAG}: Could not remove file: " + self.out_file_path) from e

    def _set_next_index(self, next_index: int):
        self._next_index: int = next_index
        self._next_range_set: set = set(range(self._next_index, self._next_index + self.chunk_size))

    def _check_closed(self):
        if self._is_closed:
            raise RuntimeError("{self.TAG}: Index buffer already is closed")

    def is_closed(self) -> bool:
        return self._is_closed

    def written_chunk_count(self) -> int:
        return self._written_chunk_count

    def size(self) -> int:
        return len(self._data)

    def insert(self, index: int, value: object):
        self._check_closed()
        if index < 0:
            raise ValueError(f"{self.TAG}: Index must be greater than or equal to 0, given: {index}")

        with self._lock:
            if DEBUG and index in self._data:
                log_warn(f"{self.TAG}: INDEX {index} already present !!!")

            self._data[index] = value
            self._consider_flush_unsafe()

    def __write_indices_to_fd(self, fd, indices_sorted, pre_string = None):
        # Pre string
        if pre_string is not None and isinstance(pre_string, str) and len(pre_string) > 0:
            fd.write(pre_string)

        # actual data
        for i in indices_sorted:
            fd.write(self.string_converter_callback(self._data.pop(i)))

    def _write_chunk_indices(self, indices_sorted, indices_size: int, execute_pre_write_callback: bool = True):
        chunk_index = self._written_chunk_count

        pre_string = None
        if execute_pre_write_callback and self.pre_chunk_write_callback is not None:
            pre_string = self.pre_chunk_write_callback(chunk_index)

        if self.keep_file_open:
            if self._out_fd is None:
                self._out_fd = open(self.out_file_path, "w")
            self.__write_indices_to_fd(self._out_fd, indices_sorted, pre_string=pre_string)
        else:
            self._consider_delete_file()
            with open(self.out_file_path, "a") as out_fd:
                self.__write_indices_to_fd(out_fd, indices_sorted, pre_string=pre_string)

        self._written_chunk_count += 1
        self._set_next_index(self._next_index + indices_size)

        # Post-write callback
        if self.post_chunk_write_callback is not None:
            self.post_chunk_write_callback(chunk_index, indices_size)

    # NOT THREAD SAFE
    def _consider_flush_unsafe(self):
        self._check_closed()

        while len(self._data) >= self.chunk_size:
            # Check if the current range exists
            range_exists = self._next_range_set.issubset(self._data.keys())
            if not range_exists:
                return

            # write chunk to file
            indices = range(self._next_index, self._next_index + self.chunk_size)
            self._write_chunk_indices(indices, indices_size=self.chunk_size)

    def close(self):
        if self._is_closed: return

        with self._lock:
            if self._is_closed: return
            self._is_closed = True

            # force flush remaining indices in order
            if len(self._data) > 0:
                sorted_indices = sorted(self._data.keys())
                self._write_chunk_indices(sorted_indices, indices_size=len(sorted_indices))
            self._data.clear()

            # Close file descriptor
            if self._out_fd is not None:
                try:
                    self._out_fd.close()
                except Exception as e:
                    log_warn(f"{self.TAG}: Failed to close output file '{self.out_file_path}' : {e}", e)


# ------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------
shutdown_event: mp.Event = mp.Event()       # MAIN SHUTDOWN EVENT. use .is_set() and .set()
SHUTDOWN_REQUESTED = False       # Only for reporting purposes

ACTIVE_RAM_FILES = set()
RAM_FILES_LOCK = threading.Lock()

index_stream_buffer: IndexStreamBuffer = None    # will initialize later
shm_buffer_main: SharedFrameBuffer = None        # will initialize later

# Time benchmarks
t_app_start = time.perf_counter()
t_init_total = 0.0
t_ram_load_total = 0.0
t_compute_total = 0.0
t_io_wait_total = 0.0

# --------------------------------------------------------------------
# Cleanup Handlers
# --------------------------------------------------------------------
def register_ram_file(filepath):
    with RAM_FILES_LOCK: ACTIVE_RAM_FILES.add(filepath)


def unregister_ram_file(filepath):
    with RAM_FILES_LOCK:
        if filepath in ACTIVE_RAM_FILES:
            ACTIVE_RAM_FILES.remove(filepath)
            if os.path.exists(filepath):
                try:
                    log_debug(f"Removing file from RAM DISK: {filepath}")
                    os.remove(filepath)
                except:
                    pass


def cleanup_ramdisk():
    with RAM_FILES_LOCK:
        if len(ACTIVE_RAM_FILES) == 0: return

        log_info(f"Cleaning up RAM disk {RAM_DISK_PATH}")
        for f in list(ACTIVE_RAM_FILES):
            if os.path.exists(f):
                try:
                    log_debug(f"Removing file from RAM DISK: {f}")
                    os.remove(f)
                except:
                    pass
            ACTIVE_RAM_FILES.remove(f)


def cleanup():
    # Safe fallback for Shared Memory cleanup if the script crashed early
    shm_buf = shm_buffer_main
    if shm_buf is not None:
        shm_buf.close()

    idx_streamer = index_stream_buffer
    if idx_streamer is not None and isinstance(idx_streamer, IndexStreamBuffer):
        log_info("Closing output file...")
        idx_streamer.close()

    cleanup_ramdisk()


# SIGNAL HANDLERS --------------------
# called on exit
def handle_exit():
    global SHUTDOWN_REQUESTED

    # CRITICAL: Prevent child processes from executing parent cleanup routines!
    if mp.current_process().name != 'MainProcess':
        return

    shutdown_event.set()
    SHUTDOWN_REQUESTED = True

    print("")
    log_info(f"Exiting...")
    cleanup()
    print("")

def handle_os_signal(signum, frame):
    global SHUTDOWN_REQUESTED

    # CRITICAL: Prevent child processes from executing parent cleanup routines!
    if mp.current_process().name != 'MainProcess':
        return

    shutdown_event.set()
    SHUTDOWN_REQUESTED = True

    sig_name = signal.Signals(signum).name
    print("")
    log_warn(f"Received OS Signal: {sig_name}. Initiating clean shutdown...")


# =============================================================================
# VALIDATION AND PRECONDITIONS
# =============================================================================
log_info(f"Starting Pair Interaction Analysis ({LABEL})")
if not SELECTION1.strip():
    log_error("SELECTION1 cannot be empty. Please define a valid atom selection.")

# Extracting boolean env vars
UPDATE_SELECTION1: bool = boolify(UPDATE_SELECTION1,
                                  err_msg="Invalid value of environment variable NAMD_ENERGY_UPDATE_SELECTION1")
UPDATE_SELECTION2: bool = boolify(UPDATE_SELECTION2,
                                  err_msg="Invalid value of environment variable NAMD_ENERGY_UPDATE_SELECTION2")

if SELECTION2.strip():
    if SELECTION1 == SELECTION2:
        log_error("SELECTION2 must be different from SELECTION1. Leave blank for self-interaction.")
    is_self_interaction: bool = False
else:
    is_self_interaction: bool = True
    UPDATE_SELECTION2: bool = False
    OUT_FORCE = False

if not OUT_FORCE: OUT_FORCE_COMPONENTS = False

if SWITCHDIST >= CUTOFF:
    log_error(f"Switching distance must be less than CUTOFF. Given Cutoff: {CUTOFF} Å, Switch dist: {SWITCHDIST} Å. Disabling switching")
    SWITCHDIST = 0.0  # disable switching

HAS_SWITCHING = SWITCHDIST > 0 and SWITCHDIST < CUTOFF

if PME_ENABLED and not PERIODIC:
    log_warn("PME only works for Periodic systems. Disabling PME...")
    PME_ENABLED = False

if FRAME_SKIP < 0:
    log_warn(f"FRAME SKIP must be >= 0. Given {FRAME_SKIP}. Resetting to 0")
    FRAME_SKIP = 0
FRAME_STEP = FRAME_SKIP + 1

if RAM_LOAD_ENABLED:
    # make sure ram disk exists
    try:
        os.makedirs(RAM_DISK_PATH, exist_ok=True)
    except Exception as e:
        log_error(f"RAM loading enabled but failed to create RAM_DISK directory {RAM_DISK_PATH}: {e}", e)

    if RAM_CHUNK_MODE and shutil.which("catdcd") is None:
        log_warn("CHUNKING DISABLED: 'catdcd' not found on system PATH. Falling back to FULL RAM LOAD [LOADER 1]...")
        RAM_CHUNK_MODE = False

# ------------------------------------------------------------------------
raw_erg_requested: set = set([e.lower().strip().lstrip('-') for e in OUT_ENERGIES])
if "all" in raw_erg_requested:
    raw_erg_requested.update(["vdw", "elec", "nonb", "pote", "total"])
    if is_self_interaction: raw_erg_requested.update(["bond", "angl", "dihe", "impr", "conf"])
if "total" in raw_erg_requested or "pote" in raw_erg_requested:
    raw_erg_requested.update(["vdw", "elec", "nonb"])
    if is_self_interaction:
        raw_erg_requested.update(["bond", "angl", "dihe", "impr", "conf"])
if "nonb" in raw_erg_requested:
    raw_erg_requested.update(["vdw", "elec"])
if "conf" in raw_erg_requested:
    raw_erg_requested.update(["bond", "angl", "dihe", "impr"])

cross_erg_supported = {"vdw", "elec", "nonb", "pote", "total", "all"}
self_erg_supported = cross_erg_supported.union({"bond", "angl", "dihe", "impr", "conf"})
actual_erg_supported = self_erg_supported if is_self_interaction else cross_erg_supported

unsupported_erg = raw_erg_requested - actual_erg_supported
if unsupported_erg:
    log_error(f"Unsupported energies requested: {unsupported_erg}")

final_erg_components: list[str] = [e for e in ["vdw", "elec", "bond", "angl", "dihe", "impr"] if e in raw_erg_requested]
if len(final_erg_components) == 0:
    log_error(f"No supported energies requested")


# =============================================================================
# OPENMM SYSTEM INITIALIZATION
# =============================================================================
log_info("Parsing Topology and Forcefield...")
psf = app.CharmmPsfFile(PSF_FILE)
params = app.CharmmParameterSet(*PARAM_FILES)

if PERIODIC:
    psf.setBox(10.0 * unit.nanometers, 10.0 * unit.nanometers, 10.0 * unit.nanometers)

base_nb_method = app.CutoffPeriodic if PERIODIC else app.CutoffNonPeriodic
base_system = psf.createSystem(params, nonbondedMethod=base_nb_method,
                               nonbondedCutoff=(CUTOFF / 10.0) * unit.nanometers)

N_ATOMS = base_system.getNumParticles()

pair_system = mm.System()
for i in range(base_system.getNumParticles()):
    pair_system.addParticle(base_system.getParticleMass(i))
if PERIODIC:
    pair_system.setDefaultPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())

# Find base nonbonded force
nb_base = [f for f in base_system.getForces() if isinstance(f, mm.NonbondedForce)][0]

# --- Coordinate Injection for Static Initialization ---
u_init = mda.Universe(PSF_FILE, DCD_FILES[0])
u_init_ts = u_init.trajectory[0]  # initialize first frame

## coordinates and box of first frame
# u_init_coords = u_init.atoms.positions
n_init_atoms = u_init.atoms.n_atoms     # num_atoms from 1 st frame
n_covalent_bonds = len(u_init.bonds)    # True covalent bonds from PSF
if N_ATOMS != n_init_atoms:
    log_error(f"Number of atoms in PSF ({N_ATOMS}) does not match number of atoms in trajectory ({u_init.atoms.n_atoms})")

if (UPDATE_SELECTION1 or UPDATE_SELECTION2) and not FAST_DYNAMIC_SELECTION:
    log_warn("FAST_DYNAMIC_SELECTION is OFF. Falling back to native MDAnalysis 'updating=True'. This may be slow!")

FAST_SEL1_OBJ = None
FAST_SEL2_OBJ = None

# Precheck & Instantiate FastDynamicSelector
if UPDATE_SELECTION1:
    if "around" not in SELECTION1.lower():
        log_warn(f"SELECTION1 ('{SELECTION1}') does not contain 'around' or 'not around'. FAST_DYNAMIC_SELECTION Disabled. You may want to turn off UPDATE_SELECTION1")
        # UPDATE_SELECTION1 = False
        FAST_DYNAMIC_SELECTION = False
    elif FAST_DYNAMIC_SELECTION:
        try:
            FAST_SEL1_OBJ = FastDynamicSelector(SELECTION1, u_init)
        except ValueError as e:
            log_error(str(e), e)

if not is_self_interaction and UPDATE_SELECTION2:
    if "around" not in SELECTION2.lower():
        log_warn(f"SELECTION2 ('{SELECTION2}') does not contain 'around' or 'not around'. FAST_DYNAMIC_SELECTION Disabled. You may want to turn off UPDATE_SELECTION2")
        # UPDATE_SELECTION2 = False
        FAST_DYNAMIC_SELECTION = False
    elif FAST_DYNAMIC_SELECTION:
        try:
            FAST_SEL2_OBJ = FastDynamicSelector(SELECTION2, u_init)
        except ValueError as e:
            log_error(str(e), e)

# Must reset this flag now
IS_DYNAMIC = UPDATE_SELECTION1 or (not is_self_interaction and UPDATE_SELECTION2)

# Frame 0 Initial Sets
if UPDATE_SELECTION1 and FAST_SEL1_OBJ is not None:
    static_sel1_idx_set = set(FAST_SEL1_OBJ.eval(u_init, PERIODIC).tolist())
else:
    sel1_init = u_init.select_atoms(SELECTION1)
    static_sel1_idx_set = set(sel1_init.indices.tolist())

if not UPDATE_SELECTION1 and len(static_sel1_idx_set) == 0:
    log_error(f"UPDATE_SELECTION1 is False, but SELECTION1 yielded 0 atoms at frame 0.")

if not is_self_interaction:
    if UPDATE_SELECTION2 and FAST_SEL2_OBJ is not None:
        static_sel2_idx_set = set(FAST_SEL2_OBJ.eval(u_init, PERIODIC).tolist()) - static_sel1_idx_set
    else:
        sel2_init = u_init.select_atoms(SELECTION2)
        static_sel2_idx_set = set(sel2_init.indices.tolist()) - static_sel1_idx_set

    if not UPDATE_SELECTION2 and len(static_sel2_idx_set) == 0:
        log_error(f"UPDATE_SELECTION2 is False, but SELECTION2 yielded 0 valid atoms at frame 0.")
else:
    static_sel2_idx_set = static_sel1_idx_set


# ------------------------------------------------------------------------
# VDW NBFIX Detection and Cloning
# ------------------------------------------------------------------------
nbfix_force = None
HAS_NBFIX = False
for f in base_system.getForces():
    if isinstance(f, mm.CustomNonbondedForce) and "acoef" in f.getEnergyFunction():
        nbfix_force = f
        HAS_NBFIX = True
        break

# ------------------------------------------------------------------------
# SYSTEM INFORMATION LOG
# ------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n------------------------------------------------------")
    print(" SYSTEM INFORMATION ")
    print("------------------------------------------------------")
    log_info(f"TOTAL ATOM COUNT: {N_ATOMS}")
    log_info(f"SELECTION-1  : \"{SELECTION1}\" (atom count at frame 0: {len(static_sel1_idx_set)})")
    log_info(f"UPDATE SEL-1 : {'ON' if UPDATE_SELECTION1 else 'OFF'}")
    if not is_self_interaction:
        log_info(f"SELECTION-2  : \"{SELECTION2}\" (atom count at frame 0: {len(static_sel2_idx_set)})")
        log_info(f"UPDATE SEL-2 : {'ON' if UPDATE_SELECTION2 else 'OFF'}")
    log_info(f"CHARMM NBFix : {'ON' if HAS_NBFIX else 'OFF'}")
    log_info(f"PERIODIC     : {'ON' if PERIODIC else 'OFF'}  (PME: {'ON' if PME_ENABLED else 'OFF'})")
    log_info(f"SWITCHING    : {'ON' if HAS_SWITCHING else 'OFF'}")
    if FRAME_STEP > 1:
        log_info(f"FRAME_STEP   : {FRAME_STEP}")
    print("------------------------------------------------------\n")

# ------------------------------------------------------------------------
# VDW and ELECTRIC Force definition
# ------------------------------------------------------------------------
# Force Group Constants
FORCE_GROUP_VDW = 1
FORCE_GROUP_ELEC = 2
FORCE_GROUP_BOND = 3
FORCE_GROUP_ANGL = 4
FORCE_GROUP_DIHE = 5
FORCE_GROUP_IMPR = 6
FORCE_GROUP_PME_RECIP = 7

# NAMD X-PLOR Cutoff & Shifting Algebra
if HAS_SWITCHING:
    ron = SWITCHDIST / 10.0
    roff = CUTOFF / 10.0
    roff2 = roff ** 2
    ron2 = ron ** 2
    denom = (roff2 - ron2) ** 3
    # NAMD X-PLOR Shift & Switch Formulas
    S_vdw = f"select(step({ron} - r), 1.0, (({roff2} - r^2)^2 * ({roff2} + 2*r^2 - 3*{ron2})) / {denom})"

    if not PERIODIC:
        S_elec = f"((1.0 - (r^2 / {roff2}))^2)"
        log_info(f"Injecting NAMD X-PLOR Switching (VDW) and Shifting (ELEC) dynamically at {SWITCHDIST}-{CUTOFF} A.")
    else:
        # S_elec = f"(1.0 - (r^2 / {roff2}))^2"
        S_elec = "1.0"
        log_info(f"Injecting NAMD X-PLOR Switching (VDW). ELEC shifting disabled (Periodic bounds detected).")
else:
    S_vdw = "1.0"
    S_elec = "1.0"

# VDW definition
if HAS_NBFIX:
    log_info("CHARMM NBFIX detected. Cloning 2D lookup tables for exact VDW.")
    vdw_base = f"(((acoef(type1, type2)/r6)^2 - bcoef(type1, type2)/r6) * {S_vdw}); r6=r^6"
else:
    log_info("Standard Lorentz-Berthelot mixing detected.")
    vdw_base = f"(4*epsilon*((sigma/r)^12 - (sigma/r)^6) * {S_vdw}); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(abs(epsilon1*epsilon2))"

# Electrostatic Definition -------------
is_elec_raw_requested = "elec" in raw_erg_requested
elec_base_main = f"({138.935456 / DIELECTRIC} * (charge1 * charge2 / r))"
pme_recip_force = None
ewald_beta = math.sqrt(-math.log(PME_TOLERANCE)) / (CUTOFF / 10.0)

if PERIODIC and PME_ENABLED:
    if PME_SHORT_RANGE_USE_EWALD_BETA:
        # Direct Space uses erfc() to perfectly blend with the Reciprocal Mesh boundary
        elec_base = f"({elec_base_main} * erfc({ewald_beta} * r))"
        log_info(f"PME Enabled. Direct-Space uses erfc() with beta = {ewald_beta:.4f} nm^-1")
    else:
        elec_base = f"({elec_base_main} * {S_elec})"

    pme_recip_force = mm.NonbondedForce()
    pme_recip_force.setNonbondedMethod(mm.NonbondedForce.PME)
    pme_recip_force.setIncludeDirectSpace(False)  # Isolate reciprocal mesh only
    pme_recip_force.setForceGroup(FORCE_GROUP_PME_RECIP)  # Group 7 (Avoids conflict with IMPR 6)

    # Static Cross uses GPU offsets for instant 0-overhead toggling
    if not IS_DYNAMIC and not is_self_interaction:
        pme_recip_force.addGlobalParameter("lambda_1", 1.0)
        pme_recip_force.addGlobalParameter("lambda_2", 1.0)
else:
    elec_base = f"({elec_base_main} * {S_elec})"
    log_info(f"Using standard Coulombic Electrostatics with Shift/Switch factor: {S_elec}")

## PARAMETER MASKING -------------------------------
print("")
if IS_DYNAMIC:
    USE_MASK = True
else:
    n1_count = len(static_sel1_idx_set)
    n2_count = len(static_sel2_idx_set) if not is_self_interaction else n1_count
    pair_count = n1_count * n2_count

    # if pairs cannot fit in RAM, use masks [SLOW but uses almost NO RAM]
    USE_MASK = pair_count > MAX_INTERACTION_PAIRS_IN_RAM

    # APPROX memory usage (only for logging)
    pairs_mb = pair_count * INTERACTION_PAIR_MEMORY_MB
    max_pairs_mb = MAX_INTERACTION_PAIRS_IN_RAM * INTERACTION_PAIR_MEMORY_MB
    if USE_MASK:
        log_warn(
            f"INTERACTION PAIR COUNT: {pair_count:,} (~{pairs_mb:.2f} MiB) exceed memory safety limit of {MAX_INTERACTION_PAIRS_IN_RAM:,} pairs (~{max_pairs_mb:.2f} MiB). Falling back to PARAMETER MASKING")
    else:
        log_warn(
            f"INTERACTION PAIR COUNT: {pair_count:,} (~{pairs_mb:.2f} MiB) fit within memory safety limit of {MAX_INTERACTION_PAIRS_IN_RAM:,} pairs (~{max_pairs_mb:.2f} MiB)")

if USE_MASK:
    log_warn("PARAMETER MASKING ENABLED: Dynamic Mode or Large Static Interaction [SLOW but uses very little memory]")
    mask_expr = "(is_sel11*is_sel12)" if is_self_interaction else "((is_sel11*is_sel22)+(is_sel21*is_sel12))"

    vdw_expr = f"mask*{vdw_base}; mask={mask_expr}"
    elec_expr = f"mask*{elec_base}; mask={mask_expr}"

    vdw_force = mm.CustomNonbondedForce(vdw_expr)
    elec_force = mm.CustomNonbondedForce(elec_expr)

    if HAS_NBFIX:
        vdw_force.addPerParticleParameter("type")
    else:
        vdw_force.addPerParticleParameter("sigma")
        vdw_force.addPerParticleParameter("epsilon")

    vdw_force.addPerParticleParameter("is_sel1")
    elec_force.addPerParticleParameter("charge")
    elec_force.addPerParticleParameter("is_sel1")

    if not is_self_interaction:
        vdw_force.addPerParticleParameter("is_sel2")
        elec_force.addPerParticleParameter("is_sel2")
else:
    log_warn("PARAMETER MASKING DISABLED. Using INTERACTION GROUPS [FAST but consumes memory]")
    vdw_force = mm.CustomNonbondedForce(vdw_base)
    elec_force = mm.CustomNonbondedForce(elec_base)

    if HAS_NBFIX:
        vdw_force.addPerParticleParameter("type")
    else:
        vdw_force.addPerParticleParameter("sigma")
        vdw_force.addPerParticleParameter("epsilon")

    elec_force.addPerParticleParameter("charge")

# Duplicate Discrete2D Tabulated Functions if NBFIX is active
if HAS_NBFIX:
    for i in range(nbfix_force.getNumTabulatedFunctions()):
        name = nbfix_force.getTabulatedFunctionName(i)
        func = nbfix_force.getTabulatedFunction(i)
        if isinstance(func, mm.Discrete2DFunction):
            nx, ny, vals = func.getFunctionParameters()
            vdw_force.addTabulatedFunction(name, mm.Discrete2DFunction(nx, ny, vals))

vdw_force.setForceGroup(FORCE_GROUP_VDW)
elec_force.setForceGroup(FORCE_GROUP_ELEC)
print("")

# ------------------------------------------------------------------------
# Setting ATOM PARAMETERS
# ------------------------------------------------------------------------
# Parameter Caches, only needed for DYNAMIC selections
dynamic_elec_q_cache_np: np.ndarray = None  # charges of all atoms, numpy type for fast math
dynamic_vdw_type_cache: np.ndarray = None       # vdw type values of each particle
dynamic_vdw_sig_eps_cache: np.ndarray = None    # vdw sigma and epsilon of each particle. 2D array [[s1,e1], [s2,e2]...]
if IS_DYNAMIC:
    if HAS_NBFIX:
        dynamic_vdw_type_cache = np.zeros(N_ATOMS, dtype=np.float64)
    else:
        dynamic_vdw_sig_eps_cache = np.zeros((N_ATOMS, 2), dtype=np.float64)

is_dynamic_elec_q_cache_needed = IS_DYNAMIC or (PME_ENABLED and is_self_interaction and is_elec_raw_requested)
if is_dynamic_elec_q_cache_needed:
    dynamic_elec_q_cache_np = np.zeros(N_ATOMS, dtype=np.float64)

for i in range(N_ATOMS):
    c, s, e = nb_base.getParticleParameters(i)
    c_val = c.value_in_unit(unit.elementary_charge)

    if HAS_NBFIX:
        type_val = nbfix_force.getParticleParameters(i)[0]
        vdw_params = (type_val,)
        if IS_DYNAMIC:
            dynamic_vdw_type_cache[i] = type_val
    else:
        s_val = s.value_in_unit(unit.nanometers)
        e_val = e.value_in_unit(unit.kilojoules_per_mole)
        vdw_params = (s_val, e_val)
        if IS_DYNAMIC:
            dynamic_vdw_sig_eps_cache[i, 0] = s_val
            dynamic_vdw_sig_eps_cache[i, 1] = e_val

    if is_dynamic_elec_q_cache_needed:
        dynamic_elec_q_cache_np[i] = c_val

    if USE_MASK:
        val1 = 1.0 if i in static_sel1_idx_set else 0.0
        val2 = 1.0 if (not is_self_interaction and i in static_sel2_idx_set) else 0.0

        if is_self_interaction:
            vdw_force.addParticle((*vdw_params, val1))
            elec_force.addParticle((c_val, val1))
        else:
            vdw_force.addParticle((*vdw_params, val1, val2))
            elec_force.addParticle((c_val, val1, val2))
    else:
        vdw_force.addParticle(vdw_params)
        elec_force.addParticle((c_val,))

    # PME Force configuration
    if PERIODIC and PME_ENABLED:
        if IS_DYNAMIC:
            # Initialize with frame 0 combined mask for A+B state
            if is_self_interaction:
                q_pme = c_val if i in static_sel1_idx_set else 0.0
            else:
                q_pme = c_val if (i in static_sel1_idx_set or i in static_sel2_idx_set) else 0.0

            # Failsafe: Prevent compiler from stripping Coulomb kernel if Frame 0 selection is completely empty
            if q_pme == 0.0 and i == 0: q_pme = 1e-10
            pme_recip_force.addParticle(q_pme, 1.0, 0.0)
        else:
            if is_self_interaction:
                q_pme = c_val if i in static_sel1_idx_set else 0.0

                # Failsafe: Prevent compiler from stripping Coulomb kernel if Frame 0 selection is completely empty
                if q_pme == 0.0 and i == 0: q_pme = 1e-10
                pme_recip_force.addParticle(q_pme, 1.0, 0.0)
            else:
                pme_recip_force.addParticle(0.0, 1.0, 0.0)
                if i in static_sel1_idx_set:
                    pme_recip_force.addParticleParameterOffset("lambda_1", i, c_val, 0.0, 0.0)
                elif i in static_sel2_idx_set:
                    pme_recip_force.addParticleParameterOffset("lambda_2", i, c_val, 0.0, 0.0)


# ------------------------------------------------------------------------
# Adding VDW and ELECTRIC forces to pair system
# ------------------------------------------------------------------------
if not USE_MASK:
    # Bipartite Small Static Interactions use InteractionGroups safely to drop water-water math natively
    vdw_force.addInteractionGroup(static_sel1_idx_set, static_sel2_idx_set)
    elec_force.addInteractionGroup(static_sel1_idx_set, static_sel2_idx_set)

nb_method = mm.CustomNonbondedForce.CutoffPeriodic if PERIODIC else mm.CustomNonbondedForce.CutoffNonPeriodic
for custom_f in [vdw_force, elec_force]:
    custom_f.setNonbondedMethod(nb_method)
    custom_f.setCutoffDistance((CUTOFF / 10.0) * unit.nanometers)
    # CRITICAL: Disable OpenMM native C5 switch because we injected NAMD X-PLOR explicitly
    custom_f.setUseSwitchingFunction(False)

pair_system.addForce(vdw_force)
pair_system.addForce(elec_force)


# -----------------------------------------------------------------------
# Exclusions & PME Exception Re-injection
# -----------------------------------------------------------------------
vdw_14_force = mm.CustomBondForce(f"4*epsilon*((sigma/r)^12 - (sigma/r)^6)")
vdw_14_force.addPerBondParameter("sigma")
vdw_14_force.addPerBondParameter("epsilon")
vdw_14_force.setUsesPeriodicBoundaryConditions(PERIODIC)
vdw_14_force.setForceGroup(FORCE_GROUP_VDW)

if PERIODIC and PME_ENABLED:
    # Subtracts artificial reciprocal mesh overlaps (erf) to yield pure scaled Coulomb math
    elec_ex_expr = f"{138.935456 / DIELECTRIC} * (q_14/r - (q_prod/r) * erf({ewald_beta}*r))"
else:
    elec_ex_expr = f"{138.935456 / DIELECTRIC} * (q_14 / r)"

elec_14_force = mm.CustomBondForce(elec_ex_expr)
elec_14_force.addPerBondParameter("q_14")
if PERIODIC and PME_ENABLED:
    elec_14_force.addPerBondParameter("q_prod")
elec_14_force.setUsesPeriodicBoundaryConditions(PERIODIC)
elec_14_force.setForceGroup(FORCE_GROUP_ELEC)

vdw_14_count, elec_ex_count = 0, 0

for i in range(nb_base.getNumExceptions()):
    p1, p2, q, s, e = nb_base.getExceptionParameters(i)
    vdw_force.addExclusion(p1, p2)
    elec_force.addExclusion(p1, p2)
    if PERIODIC and PME_ENABLED:
        pme_recip_force.addException(p1, p2, 0.0, 1.0, 0.0)

    if is_self_interaction and {p1, p2}.issubset(static_sel1_idx_set):
        q_val = q.value_in_unit(unit.elementary_charge ** 2)
        s_val = s.value_in_unit(unit.nanometers)
        e_val = e.value_in_unit(unit.kilojoules_per_mole)

        # In PME, we must subtract the mesh error even if q_14 is 0.0 (1-2 and 1-3 pairs)
        if q_val != 0.0 or (PERIODIC and PME_ENABLED):
            if PERIODIC and PME_ENABLED:
                c1 = nb_base.getParticleParameters(p1)[0].value_in_unit(unit.elementary_charge)
                c2 = nb_base.getParticleParameters(p2)[0].value_in_unit(unit.elementary_charge)
                elec_14_force.addBond(p1, p2, [q_val, c1 * c2])
            else:
                elec_14_force.addBond(p1, p2, [q_val])
            elec_ex_count += 1

        if e_val != 0.0:
            vdw_14_force.addBond(p1, p2, [s_val, e_val])
            vdw_14_count += 1

pair_system.addForce(vdw_14_force)
pair_system.addForce(elec_14_force)
if PERIODIC and PME_ENABLED:
    pair_system.addForce(pme_recip_force)
log_info(f"Re-injected Exceptions: VDW={vdw_14_count}, ELEC(PME Corrections)={elec_ex_count}")


# ------------------------------------------------------------------------
# BONDED FORCE Detection
# ------------------------------------------------------------------------
if is_self_interaction and any(e in final_erg_components for e in ["bond", "angl", "dihe", "impr", "cmap"]):
    log_info("Dynamically dispatching bonded forces from Topology...")

    for f in base_system.getForces():
        fname = type(f).__name__

        if fname == "HarmonicBondForce":
            is_ub = (f.getNumBonds() != n_covalent_bonds)  # whether this is Urey-Bradley Force

            if not is_ub and "bond" in final_erg_components:
                new_f = mm.HarmonicBondForce()
                new_f.setForceGroup(FORCE_GROUP_BOND)
                added = 0
                for i in range(f.getNumBonds()):
                    p1, p2, l, k = f.getBondParameters(i)
                    if {p1, p2}.issubset(static_sel1_idx_set):
                        new_f.addBond(p1, p2, l, k)
                        added += 1
                if added > 0:
                    pair_system.addForce(new_f)
                    log_info(f" -> Dispatched {added} Harmonic Covalent Bonds (Group 3 / BOND)")

            elif is_ub and "angl" in final_erg_components:
                new_f = mm.HarmonicBondForce()
                new_f.setForceGroup(FORCE_GROUP_ANGL)  # Route UB explicitly to ANGLE
                added = 0
                for i in range(f.getNumBonds()):
                    p1, p2, l, k = f.getBondParameters(i)
                    if {p1, p2}.issubset(static_sel1_idx_set):
                        new_f.addBond(p1, p2, l, k)
                        added += 1
                if added > 0:
                    pair_system.addForce(new_f)
                    log_info(f" -> Dispatched {added} Urey-Bradley terms (Merged into Group 4 / ANGLE)")

        elif fname == "HarmonicAngleForce" and "angl" in final_erg_components:
            new_f = mm.HarmonicAngleForce()
            new_f.setForceGroup(FORCE_GROUP_ANGL)
            added = 0
            for i in range(f.getNumAngles()):
                p1, p2, p3, th, k = f.getAngleParameters(i)
                if {p1, p2, p3}.issubset(static_sel1_idx_set):
                    new_f.addAngle(p1, p2, p3, th, k)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} Harmonic Angles (Group 4)")

        elif fname == "PeriodicTorsionForce" and "dihe" in final_erg_components:
            new_f = mm.PeriodicTorsionForce()
            new_f.setForceGroup(FORCE_GROUP_DIHE)
            added = 0
            for i in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, ph, k = f.getTorsionParameters(i)
                if {p1, p2, p3, p4}.issubset(static_sel1_idx_set):
                    new_f.addTorsion(p1, p2, p3, p4, per, ph, k)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} Periodic Torsions (Group 5)")

        elif fname == "CustomTorsionForce" and "impr" in final_erg_components:
            new_f = mm.CustomTorsionForce(f.getEnergyFunction())
            new_f.setForceGroup(FORCE_GROUP_IMPR)
            for j in range(f.getNumPerTorsionParameters()):
                new_f.addPerTorsionParameter(f.getPerTorsionParameterName(j))
            for j in range(f.getNumGlobalParameters()):
                new_f.addGlobalParameter(f.getGlobalParameterName(j), f.getGlobalParameterDefaultValue(j))
            added = 0
            for i in range(f.getNumTorsions()):
                p1, p2, p3, p4, params = f.getTorsionParameters(i)
                if {p1, p2, p3, p4}.issubset(static_sel1_idx_set):
                    new_f.addTorsion(p1, p2, p3, p4, params)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} Custom Torsions / Impropers (Group 6)")

        elif fname == "CMAPTorsionForce" and "dihe" in final_erg_components:
            new_f = mm.CMAPTorsionForce()
            new_f.setForceGroup(FORCE_GROUP_DIHE)  # Sums natively into Dihedral group
            for i in range(f.getNumMaps()):
                size, map_data = f.getMapParameters(i)
                new_f.addMap(size, map_data)
            added = 0
            for i in range(f.getNumTorsions()):
                map_idx, p1, p2, p3, p4, p5, p6, p7, p8 = f.getTorsionParameters(i)
                if {p1, p2, p3, p4, p5, p6, p7, p8}.issubset(static_sel1_idx_set):
                    new_f.addTorsion(map_idx, p1, p2, p3, p4, p5, p6, p7, p8)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} CMAP Torsions (merged into Group 5 / Dihedrals)")

    print("")



# ------------------------------------------------------------------------
# PRE-FORK MEMORY OPTIMIZATION & SYSTEM SERIALIZATION
# ------------------------------------------------------------------------
log_info("Serializing OpenMM System for independent Worker deep-copies...\n")
system_serialized_xml = mm.XmlSerializer.serialize(pair_system)

log_info("Flushing parsed topology databases before forking Compute Workers...\n")
del base_system
del psf
del params
del pair_system
del vdw_force
del elec_force
del pme_recip_force
if HAS_NBFIX: del nbfix_force

u_init.trajectory.close()
del u_init
del u_init_ts
gc.collect()  # force python gc

# Extended Group Mapping
force_group_map = {"vdw": FORCE_GROUP_VDW,
                   "elec": FORCE_GROUP_ELEC,
                   "bond": FORCE_GROUP_BOND,
                   "angl": FORCE_GROUP_ANGL,
                   "dihe": FORCE_GROUP_DIHE,
                   "impr": FORCE_GROUP_IMPR}

comp_idx = {"vdw": 0, "elec": 1, "bond": 2, "angl": 3, "dihe": 4, "impr": 5}
active_fetches = [(1 << force_group_map[c], comp_idx[c]) for c in final_erg_components]


# =============================================================================
# BACKGROUND PRODUCER PIPELINE
# =============================================================================
def parallel_reader_worker(id, shm_buffer: SharedFrameBuffer,
                           psf: str, dcd: str,
                           start: int, stop: int, step: int,
                           global_offset: int,
                           shutdown_event: mp.Event):
    u, sel1, sel2 = None, None, None
    try:
        u = mda.Universe(psf, dcd)
        # Only fallback to native MDA if our Fast Engine is disabled
        if not FAST_DYNAMIC_SELECTION or not UPDATE_SELECTION1:
            sel1 = u.select_atoms(SELECTION1, updating=UPDATE_SELECTION1)
        if not is_self_interaction and (not FAST_DYNAMIC_SELECTION or not UPDATE_SELECTION2):
            sel2 = sel1 if is_self_interaction else u.select_atoms(SELECTION2, updating=UPDATE_SELECTION2)

        # DEBUG : benchmark frame load times
        frames_read = 0
        next_prog_report_frame = PROGRESS_REPORT_INTERVAL_FRAMES
        t_last_prog_report = time.perf_counter()

        for i in range(start, stop, step):
            if shutdown_event.is_set(): break

            ts = u.trajectory[i]
            abs_f = global_offset + ts.frame

            # -------------------------------------------------
            # Execute High-Speed Distance Engine
            if FAST_DYNAMIC_SELECTION and UPDATE_SELECTION1:
                arr1 = FAST_SEL1_OBJ.eval(u, PERIODIC)
            else:
                arr1 = sel1.indices.copy()  # MDAnalysis updates it automatically when accessing indices

            if not is_self_interaction:
                if FAST_DYNAMIC_SELECTION and UPDATE_SELECTION2:
                    arr2 = FAST_SEL2_OBJ.eval(u, PERIODIC)
                else:
                    arr2 = sel2.indices.copy()  # MDAnalysis updates it automatically when accessing indices
            else:
                arr2 = None

            # ------------------------------------------------------
            # Create data copy for OpenMM
            coords_nm = u.atoms.positions / 10.0  # convert to nm for OpenMM

            # Tri-clinic box. A 3x3 matrix with unit cell vectors
            box_nm = ts.triclinic_dimensions / 10.0 if PERIODIC else None  # convert to nm for OpenMM

            # --- PHASE 2: ZERO-COPY SHARED MEMORY WRITE ---
            slot_idx = None
            while not shutdown_event.is_set():
                try:
                    # Lease a free memory block from the ring buffer
                    slot_idx = shm_buffer.free_slots.get(timeout=1.0)
                    break
                except queue.Empty:
                    continue

            if shutdown_event.is_set() or slot_idx is None:
                break

            # Dump data directly into raw RAM
            shm_buffer.write_frame(slot_idx, abs_f, coords_nm, box_nm, arr1, arr2)

            # Notify Compute Workers that this memory block is ready
            shm_buffer.ready_slots.put(slot_idx)
            # ----------------------------------------------

            ## PROGRESS TRACKER EXECUTION
            frames_read += 1
            if frames_read == next_prog_report_frame:
                t_prog_now = time.perf_counter()
                read_fps = PROGRESS_REPORT_INTERVAL_FRAMES / max(0.001, t_prog_now - t_last_prog_report)
                read_ms_per_frame = (1 / max(read_fps, 0.001)) * 1000
                log_info(f"FRAME READER {id}: Speed: {read_fps:.1f} fps  ({read_ms_per_frame:.2f} ms/frame)" + (f"  {RED}[FRAME BUFFER FULL]{NOCOL}" if shm_buffer.is_full() else ""))
                next_prog_report_frame += PROGRESS_REPORT_INTERVAL_FRAMES
                t_last_prog_report = t_prog_now
    finally:
        # Guarantee closure of internal C-level file descriptors
        if u is not None:
            try:
                u.trajectory.close()
            except:
                pass
            del u, sel1, sel2
        gc.collect()


def catdcd_chunk_loader(dcd_file: str, total_frames: int, chunk_frames: int, chunks_queue: queue.Queue, shutdown_event: mp.Event):
    global t_ram_load_total
    file_size = os.path.getsize(dcd_file)
    bytes_per_frame = file_size / total_frames if total_frames > 0 else 0
    frames_remaining = total_frames
    current_start = 1
    base_name = os.path.splitext(os.path.basename(dcd_file))[0]

    while frames_remaining > 0 and not shutdown_event.is_set():
        this_chunk_frames = min(chunk_frames, frames_remaining)
        current_last = current_start + this_chunk_frames - 1
        est_bytes = this_chunk_frames * bytes_per_frame
        margin_bytes = (RAM_SAFETY_MARGIN_GB + RAM_EXTRA_MARGIN_GB) * 1024 ** 3

        while not shutdown_event.is_set():
            free_space = shutil.disk_usage(RAM_DISK_PATH).free
            if free_space > (est_bytes + margin_bytes): break
            time.sleep(1.0)

        if shutdown_event.is_set(): break

        unique_suffix = uuid.uuid4().hex[:8]
        temp_name = os.path.join(RAM_DISK_PATH, f"{base_name}_chunk_{unique_suffix}.dcd")
        cmd = ["catdcd", "-o", temp_name, "-first", str(current_start), "-last", str(current_last), dcd_file]

        log_info(f"LOADER 2 RAM CHUNKING: Extracting frames {current_start}-{current_last} via catdcd to RAM...")
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        while proc.poll() is None:
            if shutdown_event.is_set():
                proc.terminate()
                break
            time.sleep(0.5)

        if shutdown_event.is_set():
            if os.path.exists(temp_name): os.remove(temp_name)
            break

        if proc.returncode == 0:
            t_ram_load_total += (time.perf_counter() - t0)
            register_ram_file(temp_name)
            chunks_queue.put((temp_name, this_chunk_frames))
        else:
            log_error(f"LOADER 2: catdcd subprocess failed with return code: {proc.returncode}")

        frames_remaining -= this_chunk_frames
        current_start += this_chunk_frames

    chunks_queue.put(None)


def disk_stream_blocking(shm_buffer: SharedFrameBuffer, dcd_file: str, total_frames: int, global_frame_offset: int, shutdown_event: mp.Event):
    parallel_reader_worker(1, shm_buffer, PSF_FILE, dcd_file, 0, total_frames, FRAME_STEP, global_frame_offset,
                           shutdown_event)


def ramdisk_read_blocking(shm_buffer: SharedFrameBuffer, temp_dcd: str, num_frames: int, global_frame_offset: int, shutdown_event: mp.Event):
    threads = []

    # dynamically reduce ram readers for efficiency
    reader_count = actual_ram_reader_count
    c_size = math.ceil(num_frames / reader_count)
    if c_size < RAM_READER_MIN_FRAMES:
        reader_count = math.floor(num_frames / RAM_READER_MIN_FRAMES)
        c_size = math.ceil(num_frames / reader_count)
        if reader_count != actual_ram_reader_count:
            log_warn(f"RAM READER COUNT: Dynamically reduced from {actual_ram_reader_count} => {reader_count} due to low frame count")

    for i in range(reader_count):
        start = i * c_size
        stop = min((i + 1) * c_size, num_frames)
        if start >= stop: continue
        t = threading.Thread(target=parallel_reader_worker,
                             args=(i+1, shm_buffer, PSF_FILE, temp_dcd, start, stop, FRAME_STEP, global_frame_offset,
                                   shutdown_event))
        t.daemon = True
        threads.append(t)
        t.start()

    for t in threads: t.join()


def master_producer(shm_buffer: SharedFrameBuffer, shutdown_event: mp.Event):
    global t_ram_load_total

    try:
        global_frame_offset = 0
        for dcd_file in DCD_FILES:
            if shutdown_event.is_set():
                break

            base_name = os.path.splitext(os.path.basename(dcd_file))[0]

            # Find num frames from mda.Universe
            u_temp = mda.Universe(PSF_FILE, dcd_file)
            total_frames = u_temp.trajectory.n_frames
            try:
                u_temp.trajectory.close()
            finally:
                del u_temp
                gc.collect()  # force GC

            # Logic for determining Loader Strategy
            if not RAM_LOAD_ENABLED:
                log_info(f"LOADER 0 DISK STREAM: RAM Loading Disabled. Streaming {base_name} from Disk...")
                disk_stream_blocking(shm_buffer, dcd_file, total_frames, global_frame_offset, shutdown_event)
                global_frame_offset += total_frames
                continue

            file_size = os.path.getsize(dcd_file)
            bytes_per_frame = file_size / total_frames if total_frames > 0 else 0
            free_space = shutil.disk_usage(RAM_DISK_PATH).free
            margin_bytes = (RAM_SAFETY_MARGIN_GB + RAM_EXTRA_MARGIN_GB) * 1024 ** 3
            file_fits = free_space > (file_size + margin_bytes)

            # RAM direct copy mode
            if not RAM_CHUNK_MODE or total_frames <= RAM_CHUNK_FRAMES:
                if file_fits:
                    log_info(f"LOADER 1 FULL RAM LOAD: Loading entire {os.path.basename(dcd_file)} file to RAM ...")
                    temp_dcd = os.path.join(RAM_DISK_PATH, f"{base_name}_copy_{uuid.uuid4().hex[:8]}.dcd")
                    t0 = time.perf_counter()
                    shutil.copy2(dcd_file, temp_dcd)
                    t_ram_load_total += (time.perf_counter() - t0)

                    register_ram_file(temp_dcd)
                    ramdisk_read_blocking(shm_buffer, temp_dcd, total_frames, global_frame_offset, shutdown_event)
                    unregister_ram_file(temp_dcd)
                elif not RAM_CHUNK_MODE:
                    log_warn( f"[LOADER 0] DISK STREAM: Insufficient RAM for direct copy of {os.path.basename(dcd_file)}. Falling back to disk streaming.")
                    disk_stream_blocking(shm_buffer, dcd_file, total_frames, global_frame_offset, shutdown_event)

                global_frame_offset += total_frames
                continue

            # RAM Chunk Mode: Dynamic Chunk Sizing Check (Loader 2)
            active_chunk_frames = RAM_CHUNK_FRAMES
            two_chunk_bytes = (active_chunk_frames * bytes_per_frame * 2) + margin_bytes

            if free_space < two_chunk_bytes:
                if not RAM_CHUNK_DYNAMIC:
                    log_warn(f"LOADER 2: Falling back to DISK STREAM. Insufficient RAM and Dynamic Chunking is disabled. Streaming from disk: {os.path.basename(dcd_file)}")
                    disk_stream_blocking(shm_buffer, dcd_file, total_frames, global_frame_offset, shutdown_event)
                    global_frame_offset += total_frames
                    continue

                log_warn(f"LOADER 2 LOW RAM: Cannot fit 2 default chunks ({active_chunk_frames} frames each).")
                available_for_chunks = free_space - margin_bytes
                resized_frames = int(available_for_chunks / (2 * bytes_per_frame)) if available_for_chunks > 0 else 0

                if resized_frames < RAM_CHUNK_MIN_FRAMES:
                    log_warn(f"LOADER 2: Dynamic chunk size ({resized_frames}) below MIN_CHUNK_FRAMES ({RAM_CHUNK_MIN_FRAMES}).")
                    log_warn(f"LOADER 2: Falling back to Disk Streaming for {os.path.basename(dcd_file)}.")
                    disk_stream_blocking(shm_buffer, dcd_file, total_frames, global_frame_offset, shutdown_event)
                    global_frame_offset += total_frames
                    continue

                active_chunk_frames = resized_frames
                log_info(f"LOADER 2: DYNAMIC CHUNK SIZING: Reduced chunk size to {active_chunk_frames} frames.")

            log_info(f"LOADER 2: catdcd Chunking with {active_chunk_frames} frames/chunk)")
            q_chunks = queue.Queue(maxsize=2)
            chunk_mgr_thread = threading.Thread(target=catdcd_chunk_loader,
                                                args=(
                                                dcd_file, total_frames, active_chunk_frames, q_chunks, shutdown_event))
            chunk_mgr_thread.daemon = True
            chunk_mgr_thread.start()

            chunk_frame_offset = 0
            while not shutdown_event.is_set():
                chunk_data = q_chunks.get()
                if chunk_data is None: break
                temp_dcd, n_frames = chunk_data

                ramdisk_read_blocking(shm_buffer, temp_dcd, n_frames, global_frame_offset + chunk_frame_offset,
                                      shutdown_event)
                chunk_frame_offset += n_frames
                unregister_ram_file(temp_dcd)

            chunk_mgr_thread.join()
            global_frame_offset += total_frames

    except Exception as exc:
        shutdown_event.set()
        log_error(f"Producer thread crashed: {exc}", exc)
    finally:
        # Push EOF termination tokens downstream to gracefully halt all Compute Workers
        for _ in range(NUM_COMPUTE_WORKERS):
            try:
                shm_buffer.ready_slots.put(None, timeout=1.0)
            except Exception:
                pass



# =============================================================================
# DATA FORMATTING AND OUTPUT SETUP
# =============================================================================

def create_comments_str() -> str:
    kbt_in_kcal_per_mol = 1.0 / (0.0019872 * TEMPERATURE) if TEMPERATURE is not None and TEMPERATURE > 0 else None

    comments = [
        f"Created by Python OpenMM. {get_cur_datetime_formatted()}",
        f"------------------------------------------------",
        f"=========   {LABEL}    ========",
        f"------------------------------------------------",
        f"PARAM File(s): {PARAM_FILES}",
        f"PSF File     : {PSF_FILE}",
        f"DCD File(s)  : {DCD_FILES}",
        f"TOTAL Atom Count: {N_ATOMS}",
        "## Selections ---------------",
        f"SELECTION 1  : \"{SELECTION1}\"  (atom count at frame 0: {len(static_sel1_idx_set)})",
        f"SELECTION 2  : \"{SELECTION2}\"  (atom count at frame 0: {len(static_sel2_idx_set) if not is_self_interaction else 0})",
        f"Update SEL-1 : {'ON' if UPDATE_SELECTION1 else 'OFF'}",
        f"Update SEL-2 : {'ON' if UPDATE_SELECTION2 else 'OFF'}",
        "## Output -------------------",
        f"OUTPUT Energies : {' '.join(OUT_ENERGIES)}",
        f"OUTPUT Force    : {'ON' if OUT_FORCE else 'OFF'}  (TOTAL_FORCE as VECTOR_SUM: {'ON' if TOTAL_FORCE_VECTOR_SUM else 'OFF'})",
        f"TIMESTEP First  : {TIMESTEP_FIRST} \t\t (only for book-keeping)",
        f"FRAME Frequency : {FRAME_FREQ} steps \t (only for book-keeping)",
        f"FRAME Skip      : {FRAME_SKIP} frames",
        "## Simulation Params---------",
        f"Cutoff      : {CUTOFF} Å",
        f"Switch dist : {SWITCHDIST} Å",
        f"Dielectric  : {DIELECTRIC}",
        f"PERIODIC BC : {'ON' if PERIODIC else 'OFF'}",
        f"PME         : {'ON' if PME_ENABLED else 'OFF'} (tolerance factor: {PME_TOLERANCE:.2E})",
        "-----------------------------------------------------------------",
        "NOTE-1: TOTAL energy is the same as POTENTIAL energy (interaction b/w groups)",
        "NOTE-2: TOTAL_FORCE is the VECTOR SUM of ELECT_FORCE and VDW_FORCE vectors" if TOTAL_FORCE_VECTOR_SUM else "WARNING: TOTAL_FORCE is the magnitude sum of ELECT_FORCE and VDW_FORCE magnitudes 9PHYSICALLY INACCURATE0",
        "---------",
        f"Units => ENERGY: 1 kcal/mol     = 6.95e-21 J/molecule {f'= {kbt_in_kcal_per_mol:.2e} KBT' if kbt_in_kcal_per_mol else ''}",
        "       => FORCE : 1 kcal/(mol Å) = 69.5 pN",
        "========================================================================="
    ]

    out_str = ""
    for c in comments:
        out_str += f"{COMMENT_TOKEN} {c}\n"

    return out_str


def create_output_erg_header_line() -> str:
    # Target Column Format: FRAME TS BOND ANGL DIHE IMPR CONF ELECT VDW NONBOND POTENTIAL TOTAL ELECT_FORCE VDW_FORCE TOTAL_FORCE

    headers = ["FRAME", "TS"]
    if UPDATE_SELECTION1: headers.append("SEL1_ATOM_COUNT")
    if not is_self_interaction and UPDATE_SELECTION2: headers.append("SEL2_ATOM_COUNT")

    if "bond" in raw_erg_requested: headers.append("BOND")
    if "angl" in raw_erg_requested: headers.append("ANGLE")
    if "dihe" in raw_erg_requested: headers.append("DHIED")
    if "impr" in raw_erg_requested: headers.append("IMPRP")
    if "conf" in raw_erg_requested: headers.append("CONF")
    if "elec" in raw_erg_requested: headers.append("ELECT")
    if "vdw" in raw_erg_requested: headers.append("VDW")
    if "nonb" in raw_erg_requested: headers.append("NONBOND")
    if "pote" in raw_erg_requested: headers.append("POTENTIAL")
    if "total" in raw_erg_requested: headers.append("TOTAL")

    if OUT_FORCE:
        if OUT_FORCE_COMPONENTS:
            headers.extend(["ELECT_FX", "ELECT_FY", "ELECT_FZ", "ELECT_FORCE",
                            "VDW_FX", "VDW_FY", "VDW_FZ", "VDW_FORCE"])
            if TOTAL_FORCE_VECTOR_SUM:
                headers.extend(["TOTAL_FX", "TOTAL_FY", "TOTAL_FZ", "TOTAL_FORCE"])
            else:
                headers.append("TOTAL_FORCE")
        else:
            headers.extend(["ELECT_FORCE", "VDW_FORCE", "TOTAL_FORCE"])
    return OUT_DELIMITER.join(headers) + "\n"


def create_output_erg_line(erg_row) -> str:
    abs_f_idx, n1, n2, erg_raw, f_mags, f_comps = erg_row
    ts_val = TIMESTEP_FIRST + (abs_f_idx * FRAME_FREQ)

    # energies
    e_vdw, e_elec, e_bond, e_angl, e_dihe, e_impr = erg_raw
    nonb_eng = e_vdw + e_elec
    conf_eng = e_bond + e_angl + e_dihe + e_impr
    pote_eng = nonb_eng + conf_eng

    out_row = [str(abs_f_idx), str(ts_val)]
    if UPDATE_SELECTION1: out_row.append(str(n1))
    if not is_self_interaction and UPDATE_SELECTION2: out_row.append(str(n2))

    if "bond" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(e_bond))
    if "angl" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(e_angl))
    if "dihe" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(e_dihe))
    if "impr" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(e_impr))
    if "conf" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(conf_eng))
    if "elec" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(e_elec))
    if "vdw" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(e_vdw))
    if "nonb" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(nonb_eng))
    if "pote" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(pote_eng))
    if "total" in raw_erg_requested: out_row.append(OUT_ENERGY_FORMAT.format(pote_eng))

    if OUT_FORCE and f_mags is not None:
        if OUT_FORCE_COMPONENTS and f_comps is not None:
            f_ele_mag, f_vdw_mag, f_tot_mag = f_mags

            f_ele_comp, f_vdw_comp = f_comps
            f_tot_comp = f_ele_comp + f_vdw_comp
            for i in chain(f_ele_comp, [f_ele_mag], f_vdw_comp, [f_vdw_mag], f_tot_comp, [f_tot_mag]):
                out_row.append(OUT_ENERGY_FORMAT.format(i))
        else:
            for i in f_mags:
                out_row.append(OUT_ENERGY_FORMAT.format(i))

    return OUT_DELIMITER.join(out_row) + "\n"


# =============================================================================
# INDEPENDENT COMPUTE WORKER PROCESS (Multi-processed)
# =============================================================================
def compute_worker_process(worker_id: int,
                           shutdown_event: mp.Event,
                           system_xml,
                           shm_buffer: SharedFrameBuffer,
                           out_erg_queue: mp.Queue,
                           out_meta_queue: mp.Queue | None = None):
    """
    Standalone Compute Worker Process.
    Instantiates its own OpenMM Context and reads Zero-Copy frames from shared memory.
    """
    import time
    import signal

    # Ignore OS signals in child processes; let the parent handle them cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # log_info(f"WORKER {worker_id}: Initializing OpenMM Context on [{OPENMM_PLATFORM_DISPLAY_NAME}] ....")
    local_t_start = time.perf_counter()
    local_t_total = 0.0
    local_t_init_total = 0.0
    local_t_compute_total = 0.0     # = io_wait + compute_active
    local_t_io_wait_total = 0.0

    # Deep-copy the C++ System to prevent SWIG pointer race conditions!
    local_pair_system = mm.XmlSerializer.deserialize(system_xml)

    # Extract independent force pointers from the local system
    local_vdw_force, local_elec_force, local_pme_force = None, None, None
    for f in local_pair_system.getForces():
        if f.getForceGroup() == FORCE_GROUP_VDW and isinstance(f, mm.CustomNonbondedForce):
            local_vdw_force = f
        elif f.getForceGroup() == FORCE_GROUP_ELEC and isinstance(f, mm.CustomNonbondedForce):
            local_elec_force = f
        elif f.getForceGroup() == FORCE_GROUP_PME_RECIP and isinstance(f, mm.NonbondedForce):
            local_pme_force = f

    ## Localize platform initialization to prevent CUDA Fork crashes

    # platform = OMM_PLATFORM_NAME
    # properties = {}
    # if USE_GPU:
    #     for plat_name in ['CUDA', 'HIP', 'OpenCL']:
    #         try:
    #             platform = mm.Platform.getPlatformByName(plat_name)
    #             break
    #         except Exception:
    #             continue

    # if platform is None or platform == 'CPU':
    #     platform = mm.Platform.getPlatformByName('CPU')
    #     properties = {'Threads': str(assigned_openmm_threads)}

    try:
        ## Just initialize previously detected platform
        platform = mm.Platform.getPlatformByName(OPENMM_PLATFORM_NAME)
        context = mm.Context(local_pair_system, mm.VerletIntegrator(1.0 * unit.femtoseconds), platform, OPENMM_PLATFORM_PROPERTIES)
    except Exception as exc:
        log_error(f"WORKER {worker_id}: Failed to initialize OpenMM Context on [{OPENMM_PLATFORM_DISPLAY_NAME}]: {exc}", exc)
        shutdown_event.set()
        return

    log_info(f"WORKER {worker_id}: Initialized OpenMM Context on [{OPENMM_PLATFORM_DISPLAY_NAME}]")

    # Units
    to_kcal = unit.kilocalorie_per_mole
    to_kcal_A = unit.kilocalorie_per_mole / unit.angstrom

    # Zero output cache
    zero_erg_arr = np.zeros(len(force_group_map), dtype=np.float64)
    zero_force_mags = [0.0, 0.0, 0.0] if OUT_FORCE else None
    zero_force_comps = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)] if OUT_FORCE_COMPONENTS else None

    # SWIG ACCELERATOR
    vdw_set = local_vdw_force.setParticleParameters
    elec_set = local_elec_force.setParticleParameters
    pme_set = local_pme_force.setParticleParameters if local_pme_force else None

    pme_self_prefactor = 0.0
    if PERIODIC and PME_ENABLED and is_elec_raw_requested and is_self_interaction:
        pme_self_prefactor = -(138.935456 / DIELECTRIC) * (math.sqrt(-math.log(PME_TOLERANCE)) / (CUTOFF / 10.0) / math.sqrt(math.pi))

    # HOT-LOOP OPTIMIZATION: Pre-cast static sets to NumPy arrays to bypass list() casting inside the loop
    static_arr1 = np.array(list(static_sel1_idx_set), dtype=np.int32)
    static_arr2 = np.array(list(static_sel2_idx_set), dtype=np.int32)

    prev_arr1 = static_arr1.copy() if IS_DYNAMIC else np.array([], dtype=np.int32)
    prev_arr2 = static_arr2.copy() if IS_DYNAMIC else np.array([], dtype=np.int32)
    # prev_arr1 = np.array([], dtype=np.int32)
    # prev_arr2 = np.array([], dtype=np.int32)

    # O(1) Lookup Masks
    mask1 = np.array([], dtype=bool)
    mask2 = np.array([], dtype=bool)
    if IS_DYNAMIC:
        mask1 = np.zeros(N_ATOMS, dtype=bool)
        mask2 = np.zeros(N_ATOMS, dtype=bool)

    # pre run GC
    gc.collect()

    # Counters
    local_frames_processed = 0
    next_manual_gc_frames = MANUAL_GC_INTERVAL_FRAMES

    # Time Metrics
    local_t_compute_start = time.perf_counter()
    local_t_init_total += (local_t_compute_start - local_t_start)

    while not shutdown_event.is_set():
        t_itr_start = time.perf_counter()
        try:
            slot_idx = shm_buffer.ready_slots.get(timeout=1.0)
        except queue.Empty: continue
        finally: local_t_io_wait_total += (time.perf_counter() - t_itr_start)

        if slot_idx is None: break

        # Zero-Copy Read from Shared RAM. DO NOT FREE IT YET, OTHERWISE it may get overwritten by producer
        abs_f, coords_nm, box_nm, arr_sel1, arr_sel2, n1, n2 = shm_buffer.read_frame(slot_idx)

        # Fallback to baked global static NumPy arrays if a specific selection wasn't dynamic
        curr_arr1 = arr_sel1 if arr_sel1 is not None else static_arr1
        curr_arr2 = arr_sel2 if arr_sel2 is not None else static_arr2

        if len(curr_arr1) == 0:
            log_warn(f"Worker {worker_id}: SELECTION-1 Atom count 0 at frame {abs_f}. Emitting zero-energy frame.")
            shm_buffer.free_slots.put(slot_idx)

            # Emit zero-energy string to IndexStreamBuffer to prevent contiguous deadlocks
            zero_out_str = create_output_erg_line((abs_f, 0, n2 if UPDATE_SELECTION2 else None, zero_erg_arr, zero_force_mags, zero_force_comps))

            try: out_erg_queue.put((abs_f, zero_out_str))
            except Exception: pass
            local_frames_processed += 1
            continue

        context.setPositions(coords_nm)
        if PERIODIC and box_nm is not None:
            context.setPeriodicBoxVectors(box_nm[0], box_nm[1], box_nm[2])

        actual_curr_arr2 = curr_arr2
        if IS_DYNAMIC:
            if not is_self_interaction:
                # Mathematically preserve the non-overlap rule safely using NumPy.
                # Even if sel2 is strictly static, it MUST dynamically shrink if a dynamic sel1 expands into it!
                actual_curr_arr2 = np.setdiff1d(curr_arr2, curr_arr1, assume_unique=True)

            # Update O(1) Boolean Masks instantly
            mask1.fill(False)
            mask1[curr_arr1] = True
            if not is_self_interaction:
                mask2.fill(False)
                mask2[actual_curr_arr2] = True

            # Use C-optimized XOR to find exactly which atoms changed boundary states.
            # If an array is static, its XOR against the previous frame will naturally be empty [].
            changed1 = np.setxor1d(curr_arr1, prev_arr1, assume_unique=True)

            if is_self_interaction:
                for i in changed1:
                    idx = int(i)
                    val = 1.0 if mask1[idx] else 0.0
                    q = float(dynamic_elec_q_cache_np[idx])
                    vdw_tup = (float(dynamic_vdw_type_cache[idx]), val) if HAS_NBFIX else (
                        float(dynamic_vdw_sig_eps_cache[idx, 0]), float(dynamic_vdw_sig_eps_cache[idx, 1]), val)

                    vdw_set(idx, vdw_tup)
                    elec_set(idx, (q, val))
                    if pme_set: pme_set(idx, q * val, 1.0, 0.0)
            else:
                changed2 = np.setxor1d(actual_curr_arr2, prev_arr2, assume_unique=True)
                all_changed = np.union1d(changed1, changed2)

                for i in all_changed:
                    idx = int(i)
                    val = 1.0 if mask1[idx] else 0.0
                    s2_val = 1.0 if mask2[idx] else 0.0
                    q = float(dynamic_elec_q_cache_np[idx])

                    vdw_tup = (float(dynamic_vdw_type_cache[idx]), val, s2_val) if HAS_NBFIX else (
                        float(dynamic_vdw_sig_eps_cache[idx, 0]), float(dynamic_vdw_sig_eps_cache[idx, 1]), val, s2_val)
                    vdw_set(idx, vdw_tup)
                    elec_set(idx, (q, val, s2_val))
                    if pme_set: pme_set(idx, q * max(val, s2_val), 1.0, 0.0)

            local_vdw_force.updateParametersInContext(context)
            local_elec_force.updateParametersInContext(context)
            if local_pme_force:
                local_pme_force.updateParametersInContext(context)

            # prev_arr1 = curr_arr1.copy()
            prev_arr1 = curr_arr1
            if not is_self_interaction:
                # prev_arr2 = actual_curr_arr2.copy()
                prev_arr2 = actual_curr_arr2

        # Query Energies
        erg_raw = np.zeros(len(force_group_map), dtype=np.float64)
        for mask, idx in active_fetches:
            erg_raw[idx] = context.getState(getEnergy=True, groups=mask).getPotentialEnergy().value_in_unit(to_kcal)

        # PME Subtraction block
        f_pme_cross_raw = None
        if PERIODIC and PME_ENABLED and is_elec_raw_requested:
            if is_self_interaction:
                state_recip = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << FORCE_GROUP_PME_RECIP))
                e_recip = state_recip.getPotentialEnergy().value_in_unit(to_kcal)

                sel1_q_sq_sum = np.sum(dynamic_elec_q_cache_np[curr_arr1] ** 2)
                pme_self = pme_self_prefactor * sel1_q_sq_sum
                erg_raw[comp_idx["elec"]] += (e_recip + pme_self)
                if OUT_FORCE:
                    f_pme_cross_raw = state_recip.getForces(asNumpy=True).value_in_unit(to_kcal_A)
            else:
                if IS_DYNAMIC:
                    st_AB = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << FORCE_GROUP_PME_RECIP))
                    # only_sel1 = curr_arr1[~mask2[curr_arr1]]
                    only_sel1 = curr_arr1
                    # only_sel2 = actual_curr_arr2[~mask1[actual_curr_arr2]]
                    only_sel2 = actual_curr_arr2

                    for i in only_sel2: pme_set(int(i), 0.0, 1.0, 0.0)
                    local_pme_force.updateParametersInContext(context)
                    st_A = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << FORCE_GROUP_PME_RECIP))

                    for i in only_sel2: pme_set(int(i), float(dynamic_elec_q_cache_np[i]), 1.0, 0.0)
                    for i in only_sel1: pme_set(int(i), 0.0, 1.0, 0.0)
                    local_pme_force.updateParametersInContext(context)
                    st_B = context.getState(getEnergy=True, groups=(1 << FORCE_GROUP_PME_RECIP))

                    for i in only_sel1: pme_set(int(i), float(dynamic_elec_q_cache_np[i]), 1.0, 0.0)
                    local_pme_force.updateParametersInContext(context)
                else:
                    context.setParameter("lambda_1", 1.0)
                    context.setParameter("lambda_2", 1.0)
                    st_AB = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << FORCE_GROUP_PME_RECIP))
                    context.setParameter("lambda_1", 1.0)
                    context.setParameter("lambda_2", 0.0)
                    st_A = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << FORCE_GROUP_PME_RECIP))
                    context.setParameter("lambda_1", 0.0)
                    context.setParameter("lambda_2", 1.0)
                    st_B = context.getState(getEnergy=True, groups=(1 << FORCE_GROUP_PME_RECIP))

                e_AB = st_AB.getPotentialEnergy().value_in_unit(to_kcal)
                e_A = st_A.getPotentialEnergy().value_in_unit(to_kcal)
                e_B = st_B.getPotentialEnergy().value_in_unit(to_kcal)
                erg_raw[comp_idx["elec"]] += (e_AB - e_A - e_B)

                if OUT_FORCE:
                    f_pme_cross_raw = st_AB.getForces(asNumpy=True).value_in_unit(to_kcal_A) - st_A.getForces(
                        asNumpy=True).value_in_unit(to_kcal_A)

        force_components = None
        force_mags = None
        if OUT_FORCE:
            f_ele_raw = context.getState(getForces=True, groups=(1 << FORCE_GROUP_ELEC)).getForces(asNumpy=True).value_in_unit(to_kcal_A)
            f_vdw_raw = context.getState(getForces=True, groups=(1 << FORCE_GROUP_VDW)).getForces(asNumpy=True).value_in_unit(to_kcal_A)
            if f_pme_cross_raw is not None:
                f_ele_raw += f_pme_cross_raw

            f_ele_comp = np.sum(f_ele_raw[curr_arr1], axis=0)
            f_vdw_comp = np.sum(f_vdw_raw[curr_arr1], axis=0)

            f_ele_mag = np.linalg.norm(f_ele_comp)
            f_vdw_mag = np.linalg.norm(f_vdw_comp)
            if TOTAL_FORCE_VECTOR_SUM:
                f_tot_mag = np.linalg.norm(f_ele_comp + f_vdw_comp)
            else:
                f_tot_mag = f_ele_mag + f_vdw_mag

            force_mags = [f_ele_mag, f_vdw_mag, f_tot_mag]
            if OUT_FORCE_COMPONENTS:
                force_components = [f_ele_comp, f_vdw_comp]

        # Format the output string instantly to offload string concat CPU overhead
        out_str = create_output_erg_line((abs_f,
                                          n1 if UPDATE_SELECTION1 else None,
                                          n2 if UPDATE_SELECTION2 else None,
                                          erg_raw, force_mags, force_components))

        # Push formatted string to lightweight output queue
        try:
            out_erg_queue.put((abs_f, out_str))
        except Exception: pass

        # Release the slot back to the Reader Pool
        shm_buffer.free_slots.put(slot_idx)

        # ------------------ Finalize ------------------------
        local_frames_processed += 1

        # SWIG PROXY FLUSH (DYNAMIC MEMORY STABILIZATION)
        if MANUAL_GC_ENABLED and IS_DYNAMIC and local_frames_processed == next_manual_gc_frames:
            log_debug(f"WORKER {worker_id}: Manual garbage collection at local frame {local_frames_processed}")
            gc.collect()
            next_manual_gc_frames += MANUAL_GC_INTERVAL_FRAMES

    # Final Time Metrics
    local_t_end = time.perf_counter()
    local_t_total = local_t_end - local_t_start
    local_t_compute_total = local_t_end - local_t_compute_start

    # Return local time metrics to Main process
    if out_meta_queue is not None:
        try : out_meta_queue.put((worker_id, local_t_total, local_t_init_total, local_t_compute_total, local_t_io_wait_total))
        except Exception: pass

    # Tell the Main Orchestrator this worker has gracefully terminated
    try: out_erg_queue.put(None)
    except Exception: pass

    # Worker shutdown cleanup
    try: del context
    except Exception: pass


# -----------------------------------------------------------------------------
# IndexStreamBuffer Callbacks
# -----------------------------------------------------------------------------

def _on_index_streamer_pre_chunk_write(chunk_index: int) -> str | None:
    # print(f"PRE_CHUNK_WRITE: {chunk_index}")
    if chunk_index == 0:
        return create_comments_str() + create_output_erg_header_line()
    return None


def _on_index_streamer_post_chunk_write(chunk_index: int, chunk_size: int):
    # log_debug(f"INDEX_STREAM_BUFFER: Post chunk write {chunk_index} (chunk size: {chunk_size})")
    pass


# -----------------------------------------------------------------------------
# META-DATA Processing (Returned by Workers at the end)
# -----------------------------------------------------------------------------
def process_worker_meta_data(meta_q: mp.Queue) -> np.ndarray | None:
    """
     @:returns Avg. worker times as a numpy array, or None if queue is empty
               [w_avg_total_time, w_avg_init_time, w_avg_compute_time, w_avg_io_wait_time]
     """
    w_meta_dict = {}  # worker_id -> w_meta data
    for _ in range(NUM_COMPUTE_WORKERS):
        w_meta = None
        try: w_meta = meta_q.get_nowait()
        except queue.Empty: break
        if w_meta is None: continue
        w_meta_dict[w_meta[0]] = w_meta[1:]

    n_workers = len(w_meta_dict)
    if n_workers == 0: return None

    sorted_w_ids = sorted(w_meta_dict.keys())
    w_meta_len = len(w_meta_dict[sorted_w_ids[0]])
    sum_w_t = np.zeros(w_meta_len, dtype= np.float64)

    # Process Workers meta-data in the order of their id
    for w_id in sorted_w_ids:
        w_meta_data = w_meta_dict[w_id]
        sum_w_t += w_meta_data
        if DEBUG:
            w_t_total, w_t_init, w_t_compute, w_t_io_wait = w_meta_data
            w_t_compute_active = w_t_compute - w_t_io_wait

            print("\n" + "-" * 40)
            log_debug(f"WORKER-{w_id} STATS: ")
            print(f" Total Wall Time      : {w_t_total:.1f} s")
            print(f"   ├─ Initialization  : {w_t_init:.1f} s  ({w_t_init / w_t_total * 100:.2f} %)")
            print(f"   ├─ Compute Loop    : {w_t_compute:.1f} s  ({w_t_compute / w_t_total * 100:.2f} %)")
            print(f"        ├─ Active     : {w_t_compute_active:.1f} s  ({w_t_compute_active / max(0.001, w_t_compute) * 100:.2f} %)")
            print(f"        └─ I/O Wait   : {w_t_io_wait:.1f} s  ({w_t_io_wait / max(0.001, w_t_compute) * 100:.2f} %)")
            # print("-" * 40)

    # Average worker times
    return sum_w_t / n_workers


if __name__ == '__main__':
    # Register Signal Handlers only in main process
    atexit.register(handle_exit)
    signal.signal(signal.SIGINT, handle_os_signal)
    signal.signal(signal.SIGTERM, handle_os_signal)
    try:
        signal.signal(signal.SIGHUP, handle_os_signal)
    except AttributeError:
        pass

    # Output file configuration
    out_file_path = f"{OUT_FILE_PREFIX}.energy.csv"
    index_stream_buffer = IndexStreamBuffer(output_file_path=out_file_path,
                                            chunk_size=INDEX_STREAM_BUFFER_CHUNK_SIZE,
                                            keep_file_open=INDEX_STREAM_BUFFER_ALWAYS_OPEN,
                                            string_converter_callback=lambda x: x,  # already a string
                                            pre_chunk_write_callback=_on_index_streamer_pre_chunk_write,
                                            post_chunk_write_callback=_on_index_streamer_post_chunk_write)

    # =============================================================================
    # MAIN ORCHESTRATOR & IPC CONSUMER LOOP
    # =============================================================================
    log_info(f"Allocating Zero-Copy Shared Memory Ring Buffer ({QUEUE_FRAME_COUNT} slots)...")

    # Creator allocates the physical RAM. Forked children inherit handles instantly.
    shm_buffer_main = SharedFrameBuffer(
        num_slots=QUEUE_FRAME_COUNT,
        n_atoms=N_ATOMS,
        has_dyn_sel1=UPDATE_SELECTION1,
        has_dyn_sel2=not is_self_interaction and UPDATE_SELECTION2,
        is_creator=True
    )

    # Worker energy output queue
    out_erg_q = mp.Queue()

    # Worker Meta data queue
    out_meta_q = mp.Queue()

    log_info(f"Spawning {NUM_COMPUTE_WORKERS} OpenMM Contexts (compute workers)...")
    workers = []
    for i in range(NUM_COMPUTE_WORKERS):
        p = mp.Process(target=compute_worker_process,
                       args=(i + 1, shutdown_event, system_serialized_xml, shm_buffer_main, out_erg_q, out_meta_q))
        p.start()
        workers.append(p)

    log_info("Starting Background Frame Reader Pipeline...")
    producer_process = mp.Process(target=master_producer, args=(shm_buffer_main, shutdown_event))
    producer_process.start()

    # ---------------- MAIN THREAD STRING CONSUMER ----------------
    t_compute_start = time.perf_counter()
    t_init_total += (t_compute_start - t_app_start)

    frames_processed = 0
    next_progress_report_frame = PROGRESS_REPORT_INTERVAL_FRAMES
    t_last_progress_report = t_compute_start

    active_omm_workers = NUM_COMPUTE_WORKERS
    log_info("Main Process is listening for computed frames...")
    while active_omm_workers > 0 and not shutdown_event.is_set():
        try:
            # Poll the lightweight string queue
            msg = out_erg_q.get(timeout=1.0)
        except queue.Empty:
            # Safely catch Worker crashes (e.g., OOM, Segfault)
            # If a worker dies, it won't send its 'None' token, so active_omm_workers won't decrement.
            alive_workers = sum(1 for w in workers if w.is_alive())
            if alive_workers < active_omm_workers:
                log_warn("One or more Compute Workers crashed unexpectedly. Initiating shutdown.")
                shutdown_event.set()
                SHUTDOWN_REQUESTED = True

            # Safely catch Producer crashes
            # If producer exits normally, exitcode is 0. Anything else means it crashed.
            if not producer_process.is_alive() and producer_process.exitcode not in (0, None):
                log_warn(f"Frame Producer crashed (Exit Code: {producer_process.exitcode}). Initiating shutdown.")
                shutdown_event.set()
                SHUTDOWN_REQUESTED = True
            continue

        if msg is None:
            active_omm_workers -= 1
            continue

        abs_f, out_str = msg

        # Inject directly into the ordered buffer
        index_stream_buffer.insert(index=int(abs_f / FRAME_STEP), value=out_str)
        frames_processed += 1

        # PROGRESS TRACKER EXECUTION
        if frames_processed == next_progress_report_frame:
            t_now = time.perf_counter()
            fps_current = PROGRESS_REPORT_INTERVAL_FRAMES / max(0.001, t_now - t_last_progress_report)
            fps_col = RED if fps_current < 10 else YELLOW if fps_current < 20 else GREEN
            q_frames = shm_buffer_main.size()
            q_frames_col = RED if q_frames < 3 else YELLOW if q_frames < 10 else GREEN

            log_info(f"PROGRESS: Processed {frames_processed} frames  |  Speed: {fps_col}{fps_current:.1f} fps{NOCOL}  | Queued Frames: {q_frames_col}{q_frames}{NOCOL}")
            next_progress_report_frame += PROGRESS_REPORT_INTERVAL_FRAMES
            t_last_progress_report = t_now

    log_info("Shutting down IPC pipeline...")
    shutdown_event.set()

    # Safely join processes
    producer_process.join(timeout=5)
    for w in workers:
        w.join(timeout=5)

    # Close Shared memory buffer
    shm_buffer_main.close()

    # =============================================================================
    # EXECUTION REPORT
    # =============================================================================
    t_app_end = time.perf_counter()
    t_total = t_app_end - t_app_start

    # Worker Meta Data
    worker_avg_times = process_worker_meta_data(out_meta_q)
    if worker_avg_times is not None and len(worker_avg_times) >= 4:
        t_init_total += worker_avg_times[1]
        t_compute_total += worker_avg_times[2]
        t_io_wait_total += worker_avg_times[3]
    else:
        t_compute_total = t_app_end - t_compute_start
    t_compute_active_total = max(0.0, t_compute_total - t_io_wait_total)

    avg_fps = frames_processed / max(0.001, t_compute_total)
    avg_fps_col = RED if avg_fps < 10 else YELLOW if avg_fps < 20 else GREEN
    io_wait_percent = t_io_wait_total / max(0.001, t_compute_total) * 100
    io_wait_col = RED if io_wait_percent >= 20 else YELLOW if io_wait_percent >= 10 else GREEN
    status_str = (f"{RED}ABORTED (Early Exit)" if SHUTDOWN_REQUESTED else f"{GREEN}SUCCESS") + NOCOL

    print(f"\n{get_cur_datetime_formatted()}")
    print("=" * 60)
    print(f"               EXECUTION SUMMARY")
    print("=" * 60)
    print(f" Status               : {status_str}")
    print(f" Frames Processed     : {frames_processed}")
    print(f" Processing Speed     : {avg_fps_col}{avg_fps:.1f} frames/sec{NOCOL} (avg)")
    print(f" Final Output File    : {CYAN}{out_file_path}{CYAN}")
    print(f" Compute Engine       : {OPENMM_PLATFORM_DISPLAY_NAME}")
    print("-" * 60)
    print(f" Total Wall Time      : {t_total:.1f} s")
    print(f"   ├─ Initialization  : {t_init_total:.1f} s  ({t_init_total / t_total * 100:.2f} %)")
    print(f"   ├─ catdcd/RAM I/O  : {t_ram_load_total:.1f} s  ({t_ram_load_total / t_total * 100:.2f} %)")
    print(f"   ├─ Compute Loop    : {t_compute_total:.1f} s  ({t_compute_total / t_total * 100:.2f} %)")
    print(f"        ├─ Active     : {t_compute_active_total:.1f} s  ({t_compute_active_total / max(0.001, t_compute_total) * 100:.2f} %)")
    print(f"        └─ I/O Wait   : {io_wait_col}{t_io_wait_total:.1f} s  ({io_wait_percent:.2f} %){NOCOL}")
    print("=" * 60)

    if io_wait_percent >= 20:
        log_warn("SLOW FRAME READING (i/O): You may want to enable RAM_CHUNKING or increase NUM_RAM_READERS")

    if avg_fps < 20:
        log_warn("SLOW COMPUTE PERFORMANCE: Make sure USE_GPU is enabled and increase NUM_COMPUTE_WORKERS (OpenMM Contexts)")

    print("")
