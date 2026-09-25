# ------------------------------------------------------------------------
# OpenMM and MDAnalysis implementation of NAMD PairInteraction Energy
# ------------------------------------------------------------------------
# OPTIMIZED FOR GPU's
# Much faster than cpptraj lie and VMD's namd_energy plugin


# Requirements:
# -----------------
# 1. pip packages: numpy, scipy, mdanalysis, opnemm OR openmm[cuda12] OR openmm[cuda13]
#
#    $ pip install numpy scipy mdanalysis opnemm     # OR openmm[cuda12] OR openmm[cuda13] depending on your GPU
#
# 2. external [OPTIONAL but RECOMMENDED] (should be in PATH variable)
#    => cpptraj: for chunking trajectory files to RAM
#    => catdcd : preferred for chunking DCD files to RAM


# --------------------------------------------------
# Features
# --------------------------------------------------
# => calculate interaction energy between 2 subdomains of a system from trajectory files
# => Supports both CHARMM and AMBER topologies and trajectories
# => Mimics NAMD pairinteraction feature, with similar X-PLOR based non-bonded switching
#
# => STATIC and DYNAMIC selections (update every frame, ex. hydration water)
# => Self and Cross-Interaction energies
#    -> SELECTION 1 is required
#    -> CASE 1: SELECTION 2 non-specified
#               -> calculate self-interaction energy of SELECTION 1.
#               -> Bonded (bond, angle, dihedral, improper, cross) and Non-Bonded (Elec, Vdw) energies
#
#    -> CASE 2: SELECTION 2 specified (must be different than SELECTION 1)
#               -> calculate cross-interaction energy between two selections.
#               -> Only Non-Bonded Energies (Elec, Vdw)
#
# => PME for long range electrostatics (NAMD pairinteraction does not have this)
#    Electrostatic energies with PME will be highly negative compared to NAMD pairinteraction


# --------------------------------------------------
# TRAJECTORY LOADING MODES
# --------------------------------------------------
# 1. RAM CHUNK MODE [RECOMMENDED]
#    -> Requires cpptraj (works with all trajectories) or catdcd (optional for DCDs) in PATH
#    -> Load trajectory in chunks to RAM and process frames in parallel (NUM_RAM_READERS at once)
#    -> Subsequent chunks load in the background while the current one is processing
#    -> Very fast for most cases, bypasses I/O bottlenecks from slow HDD and SSD
#    -> Smartly resize chunks to fit in RAM, best for all use-cases
#
# 2. FULL RAM LOAD (Requires large RAM)
#    -> Load Full trajectory to RAM, then start processing frames in parallel (NUM_RAM_READERS at once)
#    -> Slow loading for large trajectories
#    -> STill, much faster than reading frames directly from mechanical HDD
#
# 3. DISK STREAM (Fallback)
#    -> Read frames directly from trajectory file on disk. Almost ZERO RAM usage
#    -> Reads and processes frames one by one
#    -> Ideal for fast NVME or if your system has very low RAM
#    -> May be very slow for HDD
#
# ------------------------------------------------------------------------------
# => RECOMMENDED MODE: GPU with RAM_LOAD_ENABLED and RAM_CHUNK_MODE enabled
# ------------------------------------------------------------------------------


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
# => Template: openmm_energy.toml
#
# -> Specify parameters in this configuration file.
# -> Read comments for what each parameter does
#
# -> Some core variables can be overriden by Environment Varibles for automation with bash scripts
# -> PRIORITY ORDER: ENV_VAR > TOML config > default value in script
#
#     ENVIRONMENT VARIABLE                  VALUE             DESCRIPTION
# ----------------------------------------------------------------------------
#   -> OPENMM_ENERGY_CONFIG             =   string            config.toml file path, defaults to openmm_energy.toml
#   -> OPENMM_ENERGY_TRAJ_FILES         =   string            traj file paths separated by colon ':'
#	-> OPENMM_ENERGY_SELECTION1	        =	string            MDAnalysis selection syntax
#	-> OPENMM_ENERGY_SELECTION2	        =	string            MDAnalysis selection syntax  [OPTIONAL]
#	-> OPENMM_ENERGY_UPDATE_SELECTION1	=	boolean           true/false
#	-> OPENMM_ENERGY_UPDATE_SELECTION2	=	boolean           true/false
#	-> OPENMM_ENERGY_OUT_ENERGIES	    = 	string            output energy codes separated by space ("-elec -vdw -all")
#	-> OPENMM_ENERGY_OUT_PREFIX	        =	string            prefix of output file
#	-> OPENMM_ENERGY_LABEL              =	string			  optional label for this calculation


# --------------------------------------------------
# USAGE
# --------------------------------------------------
# 1: First run normal simulation to obtain trajectories
# 2. Copy openmm_energy.py and config TOML to working dir
# 3. Configure config TOML
# 4. [OPTIONAL] set ENVIRONMENT VARIABLES for overrides
#    -> PRIORITY ORDER: ENV_VAR > TOML config > default value in py script
#
# => RUN directly:
#    $ python3 openmm_energy.py      # defaults to "openmm_energy.toml" if present
#
# OR
#
# => RUN with CONFIG TOML file
#    $ python3 openmm_energy.py -c config.toml
#
#      OR use ENV VAR
#
#    $ export OPENMM_ENERGY_CONFIG="config.toml"
#    $ python3 openmm_energy.py
#
# OR
#
# => RUN with CONFIG + LAUNCHER script (RECOMMENDED for automating multiple runs)
# --------------------------------------------------------------------------------
#   -> Template: openmm_energy.sh
#	-> Unset selection1, selection2, out_energies, out_file_prefix in config TOML
#	-> Set environment variables in the launcher script (openmm_energy.sh)
#   => start runs
#      $ chmod +x openmm_energy.sh
#      $ ./openmm_energy.sh


# --------------------------------------------------
# OUTPUT ENERGIES
# --------------------------------------------------
# -> as sequence of 4-letter codes with '-' prefix
# => OPTIONS: -bond -angl -dihe -impr -conf -vdw -elec -nonb -pote -all
# -------------------------------------------------------------------
# -> Bonded energies     : -bond (bond), -angl (angle), -dihe (dihedral), -impr (imporper)
# -> Non-Bonded energies : -vdw (van-der waal), -elec (electrostatic)
# -> Combined energies
#    -> -conf (confomational) = bond + angle + dihedral + improper
#    -> -nonb (non-nonded)    = elec + vdw
#    -> -pote (potential)     = conf + nonb
# -------------------------------------------------------------------------------------
# => WITH SELECTION 2: ONLY [ -vdw -elec -nonb -pote -all ] ARE ALLOWED
