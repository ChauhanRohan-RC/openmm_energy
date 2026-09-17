#!/usr/bin/env python3
"""
calc_namd_energy_master.py - Production OpenMM/MDAnalysis Pair Interaction Pipeline

TODO:
1. IndexStreamBuffer: output data buffer periodically dump to file
2. Ram cache mode thread contention while reading, especially when update selection is ON

Key Architecture:
1. Smart Thread Allocation (OpenMM vs. OpenMP MDAnalysis Reader partition).
2. Hardware Bootloader: CUDA -> AMD HIP -> OpenCL -> CPU.
3. Dual RAM Loaders with Pre-Flight Anti-Deadlock Check and Dynamic Chunk Sizing.
4. Static Fast-Path Branch vs. Dynamic Differential Parameter Masking.
5. NAMD X-PLOR Shifting, NBFIX Generalized Cloning, & Dynamic Bond Dispatcher.
6. Bulletproof OS Signal Handling (SIGINT, SIGTERM, SIGHUP) & RAM Cleanup.
"""

import os
import sys
import warnings

# Suppress harmless MDAnalysis DCDReader deprecation warnings
warnings.filterwarnings("ignore", message=r".*DCDReader currently makes independent timesteps.*")

# =============================================================================
# THREAD MANAGEMENT & HARDWARE CONTROLS
# =============================================================================
USE_GPU = True  # TODO: important for smart thread allocation

# Thread controls: 0 = Smart Auto-Allocation, >0 = Explicit User Override
MDA_THREADS = 0  # Threads per MDAnalysis reader instance (OpenMP)
OPENMM_CPU_THREADS = 0  # Compute threads for OpenMM (applies only if running on CPU)
NUM_RAM_READER_THREADS = 1  # Concurrent reader threads for RAM chunks

sys_cores = os.cpu_count() or 4
actual_ram_reader_threads = min(NUM_RAM_READER_THREADS, sys_cores)

# Smart Thread Allocation Logic
if OPENMM_CPU_THREADS > 0:
    assigned_openmm_threads = OPENMM_CPU_THREADS
    openmm_alloc_mode = "User Override"
else:
    if USE_GPU:
        assigned_openmm_threads = 1
    else:
        assigned_openmm_threads = max(1, int(sys_cores * 0.75))
    openmm_alloc_mode = "Smart Auto"

if MDA_THREADS > 0:
    assigned_mda_threads = MDA_THREADS
    mda_alloc_mode = "User Override"
else:
    remaining_cores = max(1, sys_cores - assigned_openmm_threads)
    assigned_mda_threads = max(1, remaining_cores // actual_ram_reader_threads)
    mda_alloc_mode = "Smart Auto"

# Set OpenMP thread limit prior to loading C-extensions
os.environ["OMP_NUM_THREADS"] = str(assigned_mda_threads)

import gc
import time
import queue
import math
import uuid
import shutil
import signal
import atexit
import threading
from itertools import chain
import subprocess
import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit

# =============================================================================
# CONFIGURATION INPUTS
# =============================================================================

PARAM_FILES = [
    "../../common/ff/par_all36m_prot.prm",
    "../../common/ff/toppar_water_ions.prot.str"
]
PSF_FILE = "../../common/amyl_wb.psf"
DCD_FILES = ["../amyl_wb_eq2.dcd"]

# resname TIP3 and around 4.25 protein
SELECTION1: str = os.getenv("NAMD_ENERGY_SELECTION1", "")
SELECTION2: str = os.getenv("NAMD_ENERGY_SELECTION2", "")

UPDATE_SELECTION1: str = str(os.getenv("NAMD_ENERGY_UPDATE_SELECTION1", 0))
UPDATE_SELECTION2: str = str(os.getenv("NAMD_ENERGY_UPDATE_SELECTION2", 0))

OUT_FILE_PREFIX: str = os.getenv("NAMD_ENERGY_OUT_PREFIX", "interaction")
OUT_ENERGIES: list[str] = os.getenv("NAMD_ENERGY_OUT_ENERGIES", "-all").split()
LABEL: str = os.getenv("NAMD_ENERGY_LABEL", "Interaction Energy (OpenMM)")

TEMPERATURE: float = 300
CUTOFF: float = 12.0
SWITCHDIST: float = 10.0
DIELECTRIC: float = 1.0

PERIODIC: bool = True
PME_ENABLED: bool = True  # Set to True if the NAMD simulation used PME
PME_TOLERANCE: float = 1e-6  # NAMD default PME error tolerance (unitless factor)

FRAME_SKIP: int = 49
TIMESTEP_FIRST: int = 0  # only for bookkeeping
FRAME_FREQ: int = 100  # timesteps between frames (=dcd_freq). only for bookkeeping

OUT_FORCE = True
OUT_FORCE_COMPONENTS = False  # output force XYZ components
TOTAL_FORCE_VECTOR_SUM = True  # total force will be the VECTOR SUM: mag(total_force) = mag(vdw_force + elec_force vectors) [causes CANCELLATIONS, true physical behaviour].
# Else, mag(total_force) = mag(vdw_force) + mag(elec_force)    [NO CANCELLATIONS, not good physics]

OUT_DELIMITER = " "
OUT_ENERGY_FORMAT = "{:.4f}"
COMMENT_TOKEN = "#"

# --- RAM Disk & Chunk Optimization ---
RAM_LOADING_ENABLED = False
RAM_CHUNK_MODE = True
RAM_CHUNK_FRAMES = 5000  # Target frame count per chunk
DYNAMIC_CHUNK_SIZE_ENABLED = True  # Automatically shrink chunk if RAM is constrained
MIN_CHUNK_FRAMES = 2000  # Fallback to disk streaming if chunks cannot meet this size
RAM_DISK_PATH = "/tmp"
RAM_SAFETY_MARGIN_GB = 1.0  # Base free RAM margin required
RAM_EXTRA_MARGIN_GB = 0.1  # Extra buffer headroom

QUEUE_BUFFER_SIZE = 400 if USE_GPU else 100

# -----------------------------------------------------------------------------
## Other Flags ---------------------
PROGRESS_REPORT_INTERVAL_FRAMES = 10  # num frames

MANUAL_GC_ENABLED = True
MANUAL_GC_INTERVAL_FRAMES = 5000  # num frames

## Experimental Features -----------
## experimental flag to tun off erfc(ewald_beta * r) factor in short range direct electrostatics
# if true: multiplies short range raw coulomb energy with erfc(ewald_beta * r) (very fast decaying factor)
# else: uses raw coulomb energy expression for short range electrostatics with sharp discontinuity at the cutoff
PME_SHORT_RANGE_USE_EWALD_BETA = True


# =============================================================================
# HELPER FUNCTIONS & GLOBAL STATE
# =============================================================================
def log_info(msg):    print(f"\033[92m[INFO]\033[0m {msg}")


def log_warn(msg):    print(f"\033[93m[WARN]\033[0m {msg}")


def log_error(msg):   print(f"\033[91m[ERROR]\033[0m {msg}"); sys.exit(1)


# Helper functions ------------------------------

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

# -----------------------------------
# Main vars
# -----------------------------------
SHUTDOWN_REQUESTED = False
ACTIVE_RAM_FILES = set()
RAM_FILES_LOCK = threading.Lock()

t_app_start = time.perf_counter()
t_ram_load_total = 0.0
t_compute_total = 0.0
t_io_wait_total = 0.0


def register_ram_file(filepath):
    with RAM_FILES_LOCK: ACTIVE_RAM_FILES.add(filepath)


def unregister_ram_file(filepath):
    with RAM_FILES_LOCK:
        if filepath in ACTIVE_RAM_FILES:
            ACTIVE_RAM_FILES.remove(filepath)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass


def cleanup_ramdisk():
    with RAM_FILES_LOCK:
        for f in list(ACTIVE_RAM_FILES):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
            ACTIVE_RAM_FILES.remove(f)


def signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    log_warn(f"\nReceived OS Signal: {sig_name}. Initiating clean shutdown...")
    SHUTDOWN_REQUESTED = True


atexit.register(cleanup_ramdisk)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
try:
    signal.signal(signal.SIGHUP, signal_handler)
except AttributeError:
    pass

# =============================================================================
# 1. VALIDATION AND PRECONDITIONS
# =============================================================================
print("\n" + "=" * 50)
log_info(f"Starting Pair Interaction Analysis ({LABEL})")
print("=" * 50)
log_info(f"Thread Setup: OpenMM CPU={assigned_openmm_threads} ({openmm_alloc_mode}) | MDA OpenMP={assigned_mda_threads}/reader ({mda_alloc_mode})")

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

IS_DYNAMIC: bool = UPDATE_SELECTION1 or (not is_self_interaction and UPDATE_SELECTION2)

if SWITCHDIST >= CUTOFF:
    log_error(
        f"Switching distance must be less than CUTOFF. Given Cutoff: {CUTOFF} Å, Switch dist: {SWITCHDIST} Å. Disabling switching")
    SWITCHDIST = 0.0  # disable switching

if PME_ENABLED and not PERIODIC:
    log_warn("PME only works for Periodic systems. Disabling PME...")
    PME_ENABLED = False

if FRAME_SKIP < 0:
    log_warn(f"FRAME SKIP must be >= 0. Given {FRAME_SKIP}. Resetting to 0")
    FRAME_SKIP = 0
FRAME_STEP = FRAME_SKIP + 1

if RAM_LOADING_ENABLED and RAM_CHUNK_MODE:
    if shutil.which("catdcd") is None:
        log_warn("'catdcd' executable not found on system PATH. Falling back to Loader 1 (Blind Copy)...")
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

pair_system = mm.System()
for i in range(base_system.getNumParticles()):
    pair_system.addParticle(base_system.getParticleMass(i))
if PERIODIC:
    pair_system.setDefaultPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())

# Find base nonbonded force
nb_base = [f for f in base_system.getForces() if isinstance(f, mm.NonbondedForce)][0]

# --- Coordinate Injection for Static Initialization ---
u_init = mda.Universe(PSF_FILE, DCD_FILES[0])
u_init.trajectory[0]  # initialize first frame
total_atom_count = u_init.atoms.n_atoms
n_covalent_bonds = len(u_init.bonds)  # True covalent bonds from PSF

sel1_init = u_init.select_atoms(SELECTION1)
static_sel1_idx = set(sel1_init.indices.tolist())

if not UPDATE_SELECTION1 and len(static_sel1_idx) == 0:
    log_error(f"UPDATE_SELECTION1 is False, but SELECTION1 yielded 0 atoms at frame 0.")

if not is_self_interaction:
    sel2_init = u_init.select_atoms(SELECTION2)
    static_sel2_idx = set(sel2_init.indices.tolist()) - static_sel1_idx
    if not UPDATE_SELECTION2 and len(static_sel2_idx) == 0:
        log_error(f"UPDATE_SELECTION2 is False, but SELECTION2 yielded 0 valid atoms at frame 0.")
else:
    static_sel2_idx = static_sel1_idx

print("------------------------------------------------------")
log_info(f"TOTAL ATOM COUNT: {total_atom_count}")
log_info(f"SELECTION-1  : \"{SELECTION1}\" (atom count at frame 0: {len(static_sel1_idx)})")
log_info(f"UPDATE SEL-1 : {'ON' if UPDATE_SELECTION1 else 'OFF'}")
if not is_self_interaction:
    log_info(f"SELECTION-2  : \"{SELECTION2}\" (atom count at frame 0: {len(static_sel2_idx)})")
    log_info(f"UPDATE SEL-2 : {'ON' if UPDATE_SELECTION2 else 'OFF'}")
print("------------------------------------------------------")

# --- FEATURE 8: NBFIX Detection and Cloning ---
nbfix_force = None
for f in base_system.getForces():
    if isinstance(f, mm.CustomNonbondedForce) and "acoef" in f.getEnergyFunction():
        nbfix_force = f
        break

# --- FEATURE 6: NAMD X-PLOR Cutoff & Shifting Algebra ---
if SWITCHDIST > 0 and SWITCHDIST < CUTOFF:
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
if nbfix_force:
    log_info("CHARMM NBFIX detected. Cloning 2D lookup tables for exact VDW.")
    vdw_base = f"(((acoef(type1, type2)/r6)^2 - bcoef(type1, type2)/r6) * {S_vdw}); r6=r^6"
else:
    log_info("Standard Lorentz-Berthelot mixing detected.")
    vdw_base = f"(4*epsilon*((sigma/r)^12 - (sigma/r)^6) * {S_vdw}); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(abs(epsilon1*epsilon2))"

# Electrostatic Definition
is_elec_raw_requested = "elec" in raw_erg_requested
elec_base_main = f"({138.935456 / DIELECTRIC} * (charge1 * charge2 / r))"
pme_recip_force = None
ewald_beta = None

if PERIODIC and PME_ENABLED:
    ewald_beta = math.sqrt(-math.log(PME_TOLERANCE)) / (CUTOFF / 10.0)

    if PME_SHORT_RANGE_USE_EWALD_BETA:
        # Direct Space uses erfc() to perfectly blend with the Reciprocal Mesh boundary
        elec_base = f"({elec_base_main} * erfc({ewald_beta} * r))"
        log_info(f"PME Enabled. Direct-Space uses erfc() with beta = {ewald_beta:.4f} nm^-1")
    else:
        elec_base = f"({elec_base_main} * {S_elec})"

    pme_recip_force = mm.NonbondedForce()
    pme_recip_force.setNonbondedMethod(mm.NonbondedForce.PME)
    pme_recip_force.setIncludeDirectSpace(False)  # Isolate reciprocal mesh only
    pme_recip_force.setForceGroup(7)  # Group 7 (Avoids conflict with IMPR 6)

    # Static Cross uses GPU offsets for instant 0-overhead toggling
    if not IS_DYNAMIC and not is_self_interaction:
        pme_recip_force.addGlobalParameter("lambda_1", 1.0)
        pme_recip_force.addGlobalParameter("lambda_2", 1.0)
else:
    elec_base = f"({elec_base_main} * {S_elec})"
    log_info(f"Using standard Coulombic Electrostatics with Shift/Switch factor: {S_elec}")

# Optimize Memory: Force masking for Dynamic OR Static Self-Interactions to avoid OOM crashes
MAX_INTERACTION_PAIRS_IN_RAM = 250_000_000  # consumes 4 bytes per interaction pair

if IS_DYNAMIC:
    USE_MASK = True
else:
    n1_count = len(static_sel1_idx)
    n2_count = len(static_sel2_idx) if not is_self_interaction else n1_count
    pair_count = n1_count * n2_count

    if pair_count > MAX_INTERACTION_PAIRS_IN_RAM:
        log_warn(
            f"Interaction pairs ({pair_count:,}) exceed memory safety limit of {MAX_INTERACTION_PAIRS_IN_RAM:,} pairs. Falling back to Algebraic Masking. This will be slow but uses very less memory")
        USE_MASK = True
    else:
        log_info(
            f"Interaction pairs ({pair_count:,}) fit in memory. Engaging explicit InteractionGroup for max speed, but consumes RAM")
        USE_MASK = False

if USE_MASK:
    log_info("Using Parameter Masking (Dynamic Mode or Large Static Interaction).")
    mask_expr = "(is_sel11*is_sel12)" if is_self_interaction else "((is_sel11*is_sel22)+(is_sel21*is_sel12))"

    vdw_expr = f"mask*{vdw_base}; mask={mask_expr}"
    elec_expr = f"mask*{elec_base}; mask={mask_expr}"

    vdw_force = mm.CustomNonbondedForce(vdw_expr)
    elec_force = mm.CustomNonbondedForce(elec_expr)

    if nbfix_force:
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
    log_info("Masking Disabled. Engaging Static Fast-Path Native Topology.")
    vdw_force = mm.CustomNonbondedForce(vdw_base)
    elec_force = mm.CustomNonbondedForce(elec_base)

    if nbfix_force:
        vdw_force.addPerParticleParameter("type")
    else:
        vdw_force.addPerParticleParameter("sigma")
        vdw_force.addPerParticleParameter("epsilon")

    elec_force.addPerParticleParameter("charge")

# Duplicate Discrete2D Tabulated Functions if NBFIX is active
if nbfix_force:
    for i in range(nbfix_force.getNumTabulatedFunctions()):
        name = nbfix_force.getTabulatedFunctionName(i)
        func = nbfix_force.getTabulatedFunction(i)
        if isinstance(func, mm.Discrete2DFunction):
            nx, ny, vals = func.getFunctionParameters()
            vdw_force.addTabulatedFunction(name, mm.Discrete2DFunction(nx, ny, vals))

vdw_force.setForceGroup(1)
elec_force.setForceGroup(2)

# ------------------------------------------------------------------------
# Setting atom parameters
# ------------------------------------------------------------------------
N_ATOMS = base_system.getNumParticles()

# Parameter Caches, only needed for DYNAMIC selections
dynamic_elec_q_cache_np: np.ndarray = None  # charges of all atoms, numpy type for fast math
dynamic_vdw_type_cache: np.ndarray = None       # vdw type values of each particle
dynamic_vdw_sig_eps_cache: np.ndarray = None    # vdw sigma and epsilon of each particle. 2D array [[s1,e1], [s2,e2]...]
if IS_DYNAMIC:
    if nbfix_force:
        dynamic_vdw_type_cache = np.zeros(N_ATOMS, dtype=np.float64)
    else:
        dynamic_vdw_sig_eps_cache = np.zeros((N_ATOMS, 2), dtype=np.float64)

is_dynamic_elec_q_cache_needed = IS_DYNAMIC or (PME_ENABLED and is_self_interaction and is_elec_raw_requested)
if is_dynamic_elec_q_cache_needed:
    dynamic_elec_q_cache_np = np.zeros(N_ATOMS, dtype=np.float64)

for i in range(N_ATOMS):
    c, s, e = nb_base.getParticleParameters(i)
    c_val = c.value_in_unit(unit.elementary_charge)

    if nbfix_force:
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
        if IS_DYNAMIC:
            val1, val2 = 0.0, 0.0
        else:
            # Static Selections (self or cross): initializes valid masks permanently right here
            val1 = 1.0 if i in static_sel1_idx else 0.0
            val2 = 1.0 if (not is_self_interaction and i in static_sel2_idx) else 0.0

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
                q_pme = c_val if i in static_sel1_idx else 0.0
            else:
                q_pme = c_val if (i in static_sel1_idx or i in static_sel2_idx) else 0.0

            # Failsafe: Prevent compiler from stripping Coulomb kernel if Frame 0 selection is completely empty
            if q_pme == 0.0 and i == 0: q_pme = 1e-10
            pme_recip_force.addParticle(q_pme, 1.0, 0.0)
        else:
            if is_self_interaction:
                q_pme = c_val if i in static_sel1_idx else 0.0

                # Failsafe: Prevent compiler from stripping Coulomb kernel if Frame 0 selection is completely empty
                if q_pme == 0.0 and i == 0: q_pme = 1e-10
                pme_recip_force.addParticle(q_pme, 1.0, 0.0)
            else:
                pme_recip_force.addParticle(0.0, 1.0, 0.0)
                if i in static_sel1_idx:
                    pme_recip_force.addParticleParameterOffset("lambda_1", i, c_val, 0.0, 0.0)
                elif i in static_sel2_idx:
                    pme_recip_force.addParticleParameterOffset("lambda_2", i, c_val, 0.0, 0.0)

# --- FEATURE 7: Exclusions & PME Exception Re-injection ---
vdw_14_force = mm.CustomBondForce(f"4*epsilon*((sigma/r)^12 - (sigma/r)^6)")
vdw_14_force.addPerBondParameter("sigma")
vdw_14_force.addPerBondParameter("epsilon")
vdw_14_force.setUsesPeriodicBoundaryConditions(PERIODIC)
vdw_14_force.setForceGroup(1)

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
elec_14_force.setForceGroup(2)

vdw_14_count, elec_ex_count = 0, 0

for i in range(nb_base.getNumExceptions()):
    p1, p2, q, s, e = nb_base.getExceptionParameters(i)
    vdw_force.addExclusion(p1, p2)
    elec_force.addExclusion(p1, p2)
    if PERIODIC and PME_ENABLED:
        pme_recip_force.addException(p1, p2, 0.0, 1.0, 0.0)

    if is_self_interaction and {p1, p2}.issubset(static_sel1_idx):
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

if not USE_MASK:
    # Bipartite Small Static Interactions use InteractionGroups safely to drop water-water math natively
    log_warn("Using interaction groups without masking. Extremely fast but many consume RAM")
    vdw_force.addInteractionGroup(static_sel1_idx, static_sel2_idx)
    elec_force.addInteractionGroup(static_sel1_idx, static_sel2_idx)
else:
    log_info("Bypassing interaction group allocation, and using Masking. This will be slow but uses very little RAM")

nb_method = mm.CustomNonbondedForce.CutoffPeriodic if PERIODIC else mm.CustomNonbondedForce.CutoffNonPeriodic
for custom_f in [vdw_force, elec_force]:
    custom_f.setNonbondedMethod(nb_method)
    custom_f.setCutoffDistance((CUTOFF / 10.0) * unit.nanometers)
    # CRITICAL: Disable OpenMM native C5 switch because we injected NAMD X-PLOR explicitly
    custom_f.setUseSwitchingFunction(False)

pair_system.addForce(vdw_force)
pair_system.addForce(elec_force)

# --- FEATURE 5: Dynamic Bonded Force Dispatcher ---
if is_self_interaction and any(e in final_erg_components for e in ["bond", "angl", "dihe", "impr", "cmap"]):
    log_info("Dynamically dispatching bonded forces from Topology...")

    for f in base_system.getForces():
        fname = type(f).__name__

        if fname == "HarmonicBondForce":
            is_ub = (f.getNumBonds() != n_covalent_bonds)  # whether this is Urey-Bradley Force

            if not is_ub and "bond" in final_erg_components:
                new_f = mm.HarmonicBondForce()
                new_f.setForceGroup(3)
                added = 0
                for i in range(f.getNumBonds()):
                    p1, p2, l, k = f.getBondParameters(i)
                    if {p1, p2}.issubset(static_sel1_idx):
                        new_f.addBond(p1, p2, l, k)
                        added += 1
                if added > 0:
                    pair_system.addForce(new_f)
                    log_info(f" -> Dispatched {added} Harmonic Covalent Bonds (Group 3 / BOND)")

            elif is_ub and "angl" in final_erg_components:
                new_f = mm.HarmonicBondForce()
                new_f.setForceGroup(4)  # Route UB explicitly to ANGLE
                added = 0
                for i in range(f.getNumBonds()):
                    p1, p2, l, k = f.getBondParameters(i)
                    if {p1, p2}.issubset(static_sel1_idx):
                        new_f.addBond(p1, p2, l, k)
                        added += 1
                if added > 0:
                    pair_system.addForce(new_f)
                    log_info(f" -> Dispatched {added} Urey-Bradley terms (Merged into Group 4 / ANGLE)")

        elif fname == "HarmonicAngleForce" and "angl" in final_erg_components:
            new_f = mm.HarmonicAngleForce()
            new_f.setForceGroup(4)
            added = 0
            for i in range(f.getNumAngles()):
                p1, p2, p3, th, k = f.getAngleParameters(i)
                if {p1, p2, p3}.issubset(static_sel1_idx):
                    new_f.addAngle(p1, p2, p3, th, k)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} Harmonic Angles (Group 4)")

        elif fname == "PeriodicTorsionForce" and "dihe" in final_erg_components:
            new_f = mm.PeriodicTorsionForce()
            new_f.setForceGroup(5)
            added = 0
            for i in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, ph, k = f.getTorsionParameters(i)
                if {p1, p2, p3, p4}.issubset(static_sel1_idx):
                    new_f.addTorsion(p1, p2, p3, p4, per, ph, k)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} Periodic Torsions (Group 5)")

        elif fname == "CustomTorsionForce" and "impr" in final_erg_components:
            new_f = mm.CustomTorsionForce(f.getEnergyFunction())
            new_f.setForceGroup(6)
            for j in range(f.getNumPerTorsionParameters()):
                new_f.addPerTorsionParameter(f.getPerTorsionParameterName(j))
            for j in range(f.getNumGlobalParameters()):
                new_f.addGlobalParameter(f.getGlobalParameterName(j), f.getGlobalParameterDefaultValue(j))
            added = 0
            for i in range(f.getNumTorsions()):
                p1, p2, p3, p4, params = f.getTorsionParameters(i)
                if {p1, p2, p3, p4}.issubset(static_sel1_idx):
                    new_f.addTorsion(p1, p2, p3, p4, params)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} Custom Torsions / Impropers (Group 6)")

        elif fname == "CMAPTorsionForce" and "dihe" in final_erg_components:
            new_f = mm.CMAPTorsionForce()
            new_f.setForceGroup(5)  # Sums natively into Dihedral group
            for i in range(f.getNumMaps()):
                size, map_data = f.getMapParameters(i)
                new_f.addMap(size, map_data)
            added = 0
            for i in range(f.getNumTorsions()):
                map_idx, p1, p2, p3, p4, p5, p6, p7, p8 = f.getTorsionParameters(i)
                if {p1, p2, p3, p4, p5, p6, p7, p8}.issubset(static_sel1_idx):
                    new_f.addTorsion(map_idx, p1, p2, p3, p4, p5, p6, p7, p8)
                    added += 1
            if added > 0:
                pair_system.addForce(new_f)
                log_info(f" -> Dispatched {added} CMAP Torsions (merged into Group 5 / Dihedrals)")

    print("")

# Hardware Initialization (CUDA -> HIP -> OpenCL -> CPU)
platform = None
platform_name = "CPU"
properties = {}

if USE_GPU:
    for plat_name in ['CUDA', 'HIP', 'OpenCL']:
        try:
            platform = mm.Platform.getPlatformByName(plat_name)
            platform_name = f"{plat_name} (GPU)"
            break
        except Exception:
            continue
    if platform is None:
        log_warn("GPU requested but CUDA, HIP, and OpenCL are unavailable. Falling back to CPU.")

# =============================================================================
# OPENMM CONTEXT CREATION  (Memory intensive)
# =============================================================================

# --- MEMORY OPTIMIZATION: PRE-CONTEXT FLUSH ---
log_info("Flushing parsed topology databases to free RAM for Context allocation...\n")
del base_system
del psf
del params

u_init.trajectory.close()
del u_init
gc.collect()  # force python gc

log_info("Creating OpenMM Context...")
if platform is None:
    platform = mm.Platform.getPlatformByName('CPU')
    properties = {'Threads': str(assigned_openmm_threads)}
    platform_name = f"CPU ({assigned_openmm_threads} Threads)"
    context = mm.Context(pair_system, mm.VerletIntegrator(1.0 * unit.femtoseconds), platform, properties)
else:
    context = mm.Context(pair_system, mm.VerletIntegrator(1.0 * unit.femtoseconds), platform)

log_info(f"Initialized OpenMM Context on [{platform_name}]")

# Extended Group Mapping
group_map = {"vdw": 1, "elec": 2, "bond": 3, "angl": 4, "dihe": 5, "impr": 6}
comp_idx = {"vdw": 0, "elec": 1, "bond": 2, "angl": 3, "dihe": 4, "impr": 5}
active_fetches = [(1 << group_map[c], comp_idx[c]) for c in final_erg_components]


# =============================================================================
# 3. BACKGROUND PRODUCER PIPELINE
# =============================================================================
def parallel_reader_worker(q_main, psf, dcd, start, stop, global_offset, step):
    u, sel1, sel2 = None, None, None
    try:
        u = mda.Universe(psf, dcd)
        sel1 = u.select_atoms(SELECTION1, updating=UPDATE_SELECTION1)
        sel2 = sel1 if is_self_interaction else u.select_atoms(SELECTION2, updating=UPDATE_SELECTION2)

        for i in range(start, stop, step):
            if SHUTDOWN_REQUESTED: break

            ts = u.trajectory[i]
            abs_f = global_offset + ts.frame

            coords = u.atoms.positions / 10.0
            box = ts.triclinic_dimensions / 10.0 if PERIODIC else None

            arr1 = sel1.indices.copy()
            arr2 = None if is_self_interaction else sel2.indices.copy()

            while not SHUTDOWN_REQUESTED:
                try:
                    q_main.put((abs_f, coords, box, arr1, arr2), timeout=1.0)
                    break
                except queue.Full:
                    continue
    finally:
        # Guarantee closure of internal C-level file descriptors
        if u is not None:
            try:
                u.trajectory.close()
            except:
                pass
            del u, sel1, sel2
        gc.collect()


def catdcd_chunk_loader(dcd_file, total_frames, chunk_frames, q_chunks):
    global t_ram_load_total
    file_size = os.path.getsize(dcd_file)
    bytes_per_frame = file_size / total_frames if total_frames > 0 else 0
    frames_remaining = total_frames
    current_start = 1
    base_name = os.path.splitext(os.path.basename(dcd_file))[0]

    while frames_remaining > 0 and not SHUTDOWN_REQUESTED:
        this_chunk_frames = min(chunk_frames, frames_remaining)
        current_last = current_start + this_chunk_frames - 1
        est_bytes = this_chunk_frames * bytes_per_frame
        margin_bytes = (RAM_SAFETY_MARGIN_GB + RAM_EXTRA_MARGIN_GB) * 1024 ** 3

        while not SHUTDOWN_REQUESTED:
            free_space = shutil.disk_usage(RAM_DISK_PATH).free
            if free_space > (est_bytes + margin_bytes): break
            time.sleep(1.0)

        if SHUTDOWN_REQUESTED: break

        unique_suffix = uuid.uuid4().hex[:8]
        temp_name = os.path.join(RAM_DISK_PATH, f"{base_name}_chunk_{unique_suffix}.dcd")
        cmd = ["catdcd", "-o", temp_name, "-first", str(current_start), "-last", str(current_last), dcd_file]

        log_info(f"[Loader 2] Extracting frames {current_start}-{current_last} via catdcd to RAM...")
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        while proc.poll() is None:
            if SHUTDOWN_REQUESTED:
                proc.terminate()
                break
            time.sleep(0.5)

        if SHUTDOWN_REQUESTED:
            if os.path.exists(temp_name): os.remove(temp_name)
            break

        if proc.returncode == 0:
            t_ram_load_total += (time.perf_counter() - t0)
            register_ram_file(temp_name)
            q_chunks.put((temp_name, this_chunk_frames))
        else:
            log_error(f"catdcd subprocess failed with return code: {proc.returncode}")

        frames_remaining -= this_chunk_frames
        current_start += this_chunk_frames

    q_chunks.put(None)


def disk_stream(q_main, dcd_file, total_frames, global_frame_offset):
    parallel_reader_worker(q_main, PSF_FILE, dcd_file, 0, total_frames, global_frame_offset, FRAME_STEP)


def start_ramdisk_read(q_main, temp_dcd, num_frames, global_frame_offset):
    threads = []
    c_size = math.ceil(num_frames / actual_ram_reader_threads)
    for i in range(actual_ram_reader_threads):
        start = i * c_size
        stop = min((i + 1) * c_size, num_frames)
        if start >= stop: continue
        t = threading.Thread(target=parallel_reader_worker,
                             args=(q_main, PSF_FILE, temp_dcd, start, stop, global_frame_offset, FRAME_STEP))
        threads.append(t)
        t.start()

    for t in threads: t.join()


def master_producer(q_main):
    global t_ram_load_total

    try:
        global_frame_offset = 0
        for dcd_file in DCD_FILES:
            if SHUTDOWN_REQUESTED:
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
            if not RAM_LOADING_ENABLED:
                log_info(f"RAM Loading Disabled. Streaming {base_name} from Disk...")
                disk_stream(q_main, dcd_file, total_frames, global_frame_offset)
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
                    log_info(f"[Loader 1] File {os.path.basename(dcd_file)} fits in RAM. Copying blindly...")
                    temp_dcd = os.path.join(RAM_DISK_PATH, f"{base_name}_copy_{uuid.uuid4().hex[:8]}.dcd")
                    t0 = time.perf_counter()
                    shutil.copy2(dcd_file, temp_dcd)
                    t_ram_load_total += (time.perf_counter() - t0)

                    register_ram_file(temp_dcd)
                    start_ramdisk_read(q_main, temp_dcd, total_frames, global_frame_offset)
                    unregister_ram_file(temp_dcd)
                elif not RAM_CHUNK_MODE:
                    log_warn(
                        f"Insufficient RAM for direct copy of {os.path.basename(dcd_file)}. Falling back to disk streaming.")
                    disk_stream(q_main, dcd_file, total_frames, global_frame_offset)

                global_frame_offset += total_frames
                continue

            # RAM Chunk Mode: Dynamic Chunk Sizing Check (Loader 2)
            active_chunk_frames = RAM_CHUNK_FRAMES
            two_chunk_bytes = (active_chunk_frames * bytes_per_frame * 2) + margin_bytes

            if free_space < two_chunk_bytes:
                if not DYNAMIC_CHUNK_SIZE_ENABLED:
                    log_warn(
                        f"Insufficient RAM and Dynamic Chunking is disabled. Falling back to Disk Streaming for {os.path.basename(dcd_file)}.")
                    disk_stream(q_main, dcd_file, total_frames, global_frame_offset)
                    global_frame_offset += total_frames
                    continue

                log_warn(f"RAM constrained: Cannot fit 2 default chunks ({active_chunk_frames} frames each).")
                available_for_chunks = free_space - margin_bytes
                resized_frames = int(available_for_chunks / (2 * bytes_per_frame)) if available_for_chunks > 0 else 0

                if resized_frames < MIN_CHUNK_FRAMES:
                    log_warn(f"Dynamic chunk size ({resized_frames}) below MIN_CHUNK_FRAMES ({MIN_CHUNK_FRAMES}).")
                    log_warn(f"Falling back to Disk Streaming for {os.path.basename(dcd_file)}.")
                    disk_stream(q_main, dcd_file, total_frames, global_frame_offset)
                    global_frame_offset += total_frames
                    continue

                active_chunk_frames = resized_frames
                log_info(f"Dynamic Sizing: Reduced chunk size to {active_chunk_frames} frames.")

            log_info(f"Mode: LOADER 2 (catdcd Chunking with {active_chunk_frames} frames/chunk)")
            q_chunks = queue.Queue(maxsize=2)
            chunk_mgr_thread = threading.Thread(target=catdcd_chunk_loader,
                                                args=(dcd_file, total_frames, active_chunk_frames, q_chunks))
            chunk_mgr_thread.start()

            while not SHUTDOWN_REQUESTED:
                chunk_data = q_chunks.get()
                if chunk_data is None: break
                temp_dcd, n_frames = chunk_data

                start_ramdisk_read(q_main, temp_dcd, n_frames, global_frame_offset)
                unregister_ram_file(temp_dcd)

            chunk_mgr_thread.join()
            global_frame_offset += total_frames

    except Exception as e:
        log_error(f"Producer thread crashed: {e}")
    finally:
        while not SHUTDOWN_REQUESTED:
            try:
                q_main.put(None, timeout=1.0)
                break
            except queue.Full:
                continue


# =============================================================================
# 4. COMPUTE CONSUMER LOOP
# =============================================================================
frame_queue: queue.Queue = queue.Queue(maxsize=QUEUE_BUFFER_SIZE)
data_buffer = []
to_kcal = unit.kilocalorie_per_mole
to_kcal_A = unit.kilocalorie_per_mole / unit.angstrom

io_thread = threading.Thread(target=master_producer, args=(frame_queue,))
io_thread.start()

# ----- COMPUTE START ------------
# selection index trackers in dynamic mode
prev_idx_se11_set: set = set()
prev_idx_sel2_set: set = set()

frames_processed = 0
t_compute_start = time.perf_counter()
log_info(f"Compute Engine [{platform_name}] is consuming frames...")

# --- MANUAL GC SETUP ---
next_manual_gc_frames = MANUAL_GC_INTERVAL_FRAMES

# --- PROGRESS TRACKER SETUP ---
next_progress_report_frames = PROGRESS_REPORT_INTERVAL_FRAMES
t_last_progress_report = time.perf_counter()

while not SHUTDOWN_REQUESTED:
    t0_wait = time.perf_counter()
    try:
        payload = frame_queue.get(timeout=1.0)
        t_io_wait_total += (time.perf_counter() - t0_wait)
    except queue.Empty:
        t_io_wait_total += (time.perf_counter() - t0_wait)
        continue

    if payload is None: break

    abs_f, coords_nm, box_nm, arr_sel1, arr_sel2 = payload
    n1, n2 = len(arr_sel1), len(arr_sel2) if not is_self_interaction else 0
    if n1 == 0:
        log_warn(f"SELECTION-1 Atom count 0 at frame {abs_f}")
        continue

    context.setPositions(coords_nm)
    if PERIODIC:
        context.setPeriodicBoxVectors(box_nm[0], box_nm[1], box_nm[2])

    idx_set1_set: set = None
    idx_set2_set: set = None
    if IS_DYNAMIC:
        idx_set1_set: set = set(arr_sel1)
        idx_set2_set: set = idx_set1_set if is_self_interaction else (set(arr_sel2) - idx_set1_set)

        union_idx_sel1: set = idx_set1_set ^ prev_idx_se11_set
        union_idx_sel2: set = idx_set2_set ^ prev_idx_sel2_set

        for i in union_idx_sel1:
            val = 1.0 if i in idx_set1_set else 0.0
            q = float(dynamic_elec_q_cache_np[i])
            vdw_tup = (dynamic_vdw_type_cache[i], val) if nbfix_force else (dynamic_vdw_sig_eps_cache[i, 0], dynamic_vdw_sig_eps_cache[i, 1], val)

            if is_self_interaction:
                vdw_force.setParticleParameters(i, vdw_tup)
                elec_force.setParticleParameters(i, (q, val))
                if PERIODIC and PME_ENABLED:
                    pme_recip_force.setParticleParameters(i, q * val, 1.0, 0.0)
            else:
                s2_val = 1.0 if i in idx_set2_set else 0.0
                vdw_force.setParticleParameters(i, vdw_tup + (s2_val, ))
                elec_force.setParticleParameters(i, (q, val, s2_val))
                if PERIODIC and PME_ENABLED:
                    pme_recip_force.setParticleParameters(i, q * max(val, s2_val), 1.0, 0.0)

        if not is_self_interaction:
            for i in union_idx_sel2.difference(union_idx_sel1):
                q = float(dynamic_elec_q_cache_np[i])
                s2_val = 1.0 if i in idx_set2_set else 0.0
                s1_val = 1.0 if i in idx_set1_set else 0.0

                vdw_tup = (dynamic_vdw_type_cache[i], s1_val, s2_val) if nbfix_force else (dynamic_vdw_sig_eps_cache[i, 0], dynamic_vdw_sig_eps_cache[i, 1], s1_val, s2_val)
                vdw_force.setParticleParameters(i, vdw_tup)
                elec_force.setParticleParameters(i, (q, s1_val, s2_val))
                if PERIODIC and PME_ENABLED:
                    pme_recip_force.setParticleParameters(i, q * max(s1_val, s2_val), 1.0, 0.0)

        vdw_force.updateParametersInContext(context)
        elec_force.updateParametersInContext(context)
        if PERIODIC and PME_ENABLED:
            pme_recip_force.updateParametersInContext(context)

        prev_idx_se11_set = idx_set1_set
        prev_idx_sel2_set = idx_set2_set

    # Query Energies
    erg_raw = [0.0] * len(group_map)  # 6
    for mask, idx in active_fetches:
        erg_raw[idx] = context.getState(getEnergy=True, groups=mask).getPotentialEnergy().value_in_unit(to_kcal)

    # --- PME Reciprocal Space 3-Pass Subtraction ---
    f_pme_cross_raw = None
    if PERIODIC and PME_ENABLED and is_elec_raw_requested:
        if is_self_interaction:
            state_recip = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << 7))
            e_recip = state_recip.getPotentialEnergy().value_in_unit(to_kcal)

            if is_dynamic_elec_q_cache_needed:  # if we have cache
                sel1_q_sq_sum = np.sum(dynamic_elec_q_cache_np[arr_sel1] ** 2)
            else:
                sel1_q_sq_sum = 0
                for i in arr_sel1:
                    sel1_q_sq_sum += nb_base.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) ** 2

            pme_self = - (138.935456 / DIELECTRIC) * (ewald_beta / math.sqrt(math.pi)) * sel1_q_sq_sum
            erg_raw[comp_idx["elec"]] += (e_recip + pme_self)
            if OUT_FORCE:
                f_pme_cross_raw = state_recip.getForces(asNumpy=True).value_in_unit(to_kcal_A)
        else:
            if IS_DYNAMIC:
                st_AB = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << 7))
                only_sel1 = idx_set1_set - idx_set2_set
                only_sel2 = idx_set2_set - idx_set1_set

                # Pass A:
                for i in only_sel2:
                    pme_recip_force.setParticleParameters(i, 0.0, 1.0, 0.0)
                pme_recip_force.updateParametersInContext(context)
                st_A = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << 7))

                # Pass B:
                for i in only_sel2:
                    pme_recip_force.setParticleParameters(i, dynamic_elec_q_cache_np[i], 1.0, 0.0)
                for i in only_sel1:
                    pme_recip_force.setParticleParameters(i, 0.0, 1.0, 0.0)
                pme_recip_force.updateParametersInContext(context)
                st_B = context.getState(getEnergy=True, groups=(1 << 7))

                # Restore XOR AB state for next frame:
                for i in only_sel1:
                    pme_recip_force.setParticleParameters(i, dynamic_elec_q_cache_np[i], 1.0, 0.0)
                pme_recip_force.updateParametersInContext(context)
            else:
                context.setParameter("lambda_1", 1.0)
                context.setParameter("lambda_2", 1.0)
                st_AB = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << 7))
                context.setParameter("lambda_1", 1.0)
                context.setParameter("lambda_2", 0.0)
                st_A = context.getState(getEnergy=True, getForces=OUT_FORCE, groups=(1 << 7))
                context.setParameter("lambda_1", 0.0)
                context.setParameter("lambda_2", 1.0)
                st_B = context.getState(getEnergy=True, groups=(1 << 7))

            e_AB = st_AB.getPotentialEnergy().value_in_unit(to_kcal)
            e_A = st_A.getPotentialEnergy().value_in_unit(to_kcal)
            e_B = st_B.getPotentialEnergy().value_in_unit(to_kcal)
            erg_raw[comp_idx["elec"]] += (e_AB - e_A - e_B)  # PME self-energy perfectly cancels algebraically here!

            if OUT_FORCE:
                f_pme_cross_raw = st_AB.getForces(asNumpy=True).value_in_unit(to_kcal_A) - st_A.getForces(
                    asNumpy=True).value_in_unit(to_kcal_A)

    # Query Forces
    force_components = None  # [elec_force_components, vdw_force_components]  # [elec_force_components, vdw_force_components]
    force_mags = None  # [mag(elec_force), mag(vdw_force), mag(total_force)]
    if OUT_FORCE:
        f_ele_raw = context.getState(getForces=True, groups=(1 << 2)).getForces(asNumpy=True).value_in_unit(to_kcal_A)
        f_vdw_raw = context.getState(getForces=True, groups=(1 << 1)).getForces(asNumpy=True).value_in_unit(to_kcal_A)
        if f_pme_cross_raw is not None:
            f_ele_raw += f_pme_cross_raw
        f_ele_comp = np.sum(f_ele_raw[arr_sel1], axis=0)
        f_vdw_comp = np.sum(f_vdw_raw[arr_sel1], axis=0)

        f_ele_mag = np.linalg.norm(f_ele_comp)
        f_vdw_mag = np.linalg.norm(f_vdw_comp)
        if TOTAL_FORCE_VECTOR_SUM:
            # vector sum. PHYSICALLY ACCURATE
            f_tot_mag = np.linalg.norm(f_ele_comp + f_vdw_comp)
        else:
            # NOTE: sum of magnitudes. NOT PHYSICALLY ACCURATE
            f_tot_mag = f_ele_mag + f_vdw_mag

        force_mags = [f_ele_mag, f_vdw_mag, f_tot_mag]
        if OUT_FORCE_COMPONENTS:
            force_components = [f_ele_comp, f_vdw_comp]

    # final data
    data_buffer.append((abs_f,
                        n1 if UPDATE_SELECTION1 else None,
                        n2 if UPDATE_SELECTION2 else None,
                        erg_raw,
                        force_mags,
                        force_components))

    # ------------------ Finalize ------------------------
    frames_processed += 1

    # SWIG PROXY FLUSH (DYNAMIC MEMORY STABILIZATION)
    if MANUAL_GC_ENABLED and IS_DYNAMIC and frames_processed == next_manual_gc_frames:
        gc.collect()
        next_manual_gc_frames += MANUAL_GC_INTERVAL_FRAMES

    # PROGRESS TRACKER EXECUTION
    if frames_processed == next_progress_report_frames:
        t_now = time.perf_counter()
        fps_current = PROGRESS_REPORT_INTERVAL_FRAMES / max(0.001, t_now - t_last_progress_report)
        log_info(
            f"Progress: Processed {frames_processed} frames  |  Speed: {fps_current:.1f} fps  |  Queued Frames: {frame_queue.qsize()}")
        next_progress_report_frames += PROGRESS_REPORT_INTERVAL_FRAMES
        t_last_progress_report = t_now

t_compute_total = time.perf_counter() - t_compute_start
io_thread.join()

# =============================================================================
# 5. POST-LOOP FORMATTING AND OUTPUT
# =============================================================================

def create_comments_str() -> str:
    kbt_val = 1.0 / (0.0019872 * TEMPERATURE)

    comments = [
        f"================ {LABEL} ================",
        f"PARAM File(s): {PARAM_FILES}",
        f"PSF File     : {PSF_FILE}",
        f"DCD File(s)  : {DCD_FILES}",
        f"TOTAL Atom Count: {total_atom_count}",
        "## Selections ---------------",
        f"SELECTION 1  : \"{SELECTION1}\"  (atom count at frame 0: {len(static_sel1_idx)})",
        f"SELECTION 2  : \"{SELECTION2}\"  (atom count at frame 0: {len(static_sel2_idx) if not is_self_interaction else 0})",
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
        f"Switchdist  : {SWITCHDIST} Å",
        f"Dielectric  : {DIELECTRIC}",
        f"PERIODIC BC : {'ON' if PERIODIC else 'OFF'}",
        f"PME         : {'ON' if PME_ENABLED else 'OFF'} (tolerance factor: {PME_TOLERANCE:.2E})",
        "-----------------------------------------------------------------",
        "TOTAL_FORCE is the VECTOR SUM of ELECT_FORCE and VDW_FORCE vectors" if TOTAL_FORCE_VECTOR_SUM else "WARNING: TOTAL_FORCE is the magnitude sum of ELECT_FORCE and VDW_FORCE magnitudes 9PHYSICALLY INACCURATE0",
        f"Units => ENERGY: 1 kcal/mol     = 6.95e-21 J/molecule = {kbt_val:.2e} KBT",
        "       => FORCE : 1 kcal/(mol Å) = 69.5 pN",
        f"Created by Python OpenMM. {get_cur_datetime_formatted()}",
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

# -----------------------------------------------
# Writing output to file
# -----------------------------------------------
data_buffer.sort(key=lambda x: x[0])

output_file = f"{OUT_FILE_PREFIX}.energy4.csv"      # TODO: TEST
with open(output_file, 'w') as f_out:
    f_out.write(create_comments_str()) # Comments
    f_out.write(create_output_erg_header_line())  # energy header
    # energies for each frame
    for row in data_buffer:
        f_out.write(create_output_erg_line(row))

# =============================================================================
# 6. EXECUTION REPORT
# =============================================================================
t_total = time.perf_counter() - t_app_start
compute_active_time = max(0.0, t_compute_total - t_io_wait_total)
fps = frames_processed / max(0.001, t_compute_total)
status_str = "\033[91mABORTED (Early Exit)\033[0m" if SHUTDOWN_REQUESTED else "\033[92mSUCCESS\033[0m"

print(f"\n{get_cur_datetime_formatted()}")
print("=" * 60)
print(f"               EXECUTION SUMMARY")
print("=" * 60)
print(f" Status               : {status_str}")
print(f" Frames Processed     : {frames_processed}")
print(f" Processing Speed     : {fps:.1f} frames/sec")
print(f" Compute Engine       : {platform_name}")
print(f" Final Output File    : {output_file}")
print("-" * 60)
print(f" Total Wall Time      : {t_total:.1f} s")
print(f"   ├─ catdcd/RAM I/O  : {t_ram_load_total:.1f} s")
print(f"   ├─ Compute Loop    : {t_compute_total:.1f} s")
print(f"   │    ├─ Active     : {compute_active_time:.1f} s")
print(f"   │    └─ I/O Wait   : {t_io_wait_total:.1f} s")
print("=" * 60 + "\n")
