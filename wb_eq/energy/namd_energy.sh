#!/bin/bash

##############################################
# Energy Calculations using namd_energy.tcl (static selections only)
##############################################
# TODO: set env vars for each run

# Exit on error
#set -e

module load namd3 vmd

# For STATIC SELECTIONS (that do not change with time)
run_namd_energy_static() {
	export NAMD_ENERGY_PROCESSES=$NAMD_ENERGY_PROCESSES_STATIC
	export VMDNOCUDA="on"
	vmd -dispdev text -e namd_energy.static.tcl
	unset VMDNOCUDA
	unset NAMD_ENERGY_PROCESSES
}

# For DYNAMIC SELECTIONS (that change with time)
run_namd_energy_dynamic() {
	export NAMD_ENERGY_PROCESSES=$NAMD_ENERGY_PROCESSES_DYNAMIC
	export VMDNOCUDA="on"
	vmd -dispdev text -e namd_energy.dynamic.tcl
	unset VMDNOCUDA
	unset NAMD_ENERGY_PROCESSES
}


# 2nd HYDRATION SHELL CUTOFF (from RDF vs r plot minima)
export selection_water_hydration="water and (within 4.25 of protein)"
export selection_water_bulk="water and not (within 5.0 of protein)"

export NAMD_ENERGY_PROCESSES_STATIC=20
export NAMD_ENERGY_PROCESSES_DYNAMIC=30

# TEST
OUT_FILE_SUFFIX="8"

# Run 1: Protein Self
export NAMD_ENERGY_LABEL="Protein Self-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_OUT_PREFIX="prot_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-conf -nonb -kine -pote"
run_namd_energy_static

# Run 2: Water Self
export NAMD_ENERGY_LABEL="Water Self-Interaction Energy"
export NAMD_ENERGY_SELECTION1="water"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_OUT_PREFIX="water_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -kine -pote"
run_namd_energy_static

# Run 3: Protein-Water
export NAMD_ENERGY_LABEL="Protein-Water Cross-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2="water"
export NAMD_ENERGY_OUT_PREFIX="prot_water${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_namd_energy_static

#-----------------------------------------------------------------

# Run: Hydration Water Self
export NAMD_ENERGY_LABEL="Hydration Water Self-interaction Energy"
export NAMD_ENERGY_SELECTION1="$selection_water_hydration"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_OUT_PREFIX="water_hydration_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -kine -pote"
run_namd_energy_dynamic

# # Run: Bulk Water Self
# export NAMD_ENERGY_LABEL="Bulk Water Self-interaction Energy"
# export NAMD_ENERGY_SELECTION1="$selection_water_bulk"
# export NAMD_ENERGY_SELECTION2=""
# export NAMD_ENERGY_OUT_PREFIX="water_bulk_self${OUT_FILE_SUFFIX}"
# export NAMD_ENERGY_OUT_ENERGIES="-nonb -kine -pote"
# run_namd_energy_dynamic
#
#
## Run: Protein-Hydration Water
export NAMD_ENERGY_LABEL="Protein Hydration Water Cross-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2="$selection_water_hydration"
export NAMD_ENERGY_OUT_PREFIX="prot_water_hydration${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-all"
run_namd_energy_dynamic


# Run: Protein-Bulk Water
export NAMD_ENERGY_LABEL="Protein Bulk Water Cross-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2="$selection_water_bulk"
export NAMD_ENERGY_OUT_PREFIX="prot_water_bulk${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_namd_energy_dynamic
