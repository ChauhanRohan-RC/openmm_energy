#!/bin/bash

#######################################################
# Energy Calculations using MDAnalysis and OpenMM (namd_energy_openmm.py)
#######################################################
# TODO: set env vars for each run

# Exit on error
#set -e

run_openmm() {
	python3 namd_energy_openmm.py
}

run_openmm_static() {
	export NAMD_ENERGY_UPDATE_SELECTION1=0
	export NAMD_ENERGY_UPDATE_SELECTION2=0
	run_openmm
	unset NAMD_ENERGY_UPDATE_SELECTION1
	unset NAMD_ENERGY_UPDATE_SELECTION2
}


# 2nd HYDRATION SHELL CUTOFF (from RDF vs r plot minima)
# TODO: use mdanalysis selection syntax
export selection_water_hydration="water and around 4.25 protein"
export selection_water_bulk="water and not around 5.0 protein"

# TEST
OUT_FILE_SUFFIX="7"


# Run 1: Protein Self
export NAMD_ENERGY_LABEL="Protein Self-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_OUT_PREFIX="prot_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-conf -nonb -pote"
run_openmm_static

## Run 2: Water Self
export NAMD_ENERGY_LABEL="Water Self-Interaction Energy"
export NAMD_ENERGY_SELECTION1="water"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_OUT_PREFIX="water_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm_static

## Run 3: Protein-Water
export NAMD_ENERGY_LABEL="Protein-Water Cross-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2="water"
export NAMD_ENERGY_OUT_PREFIX="prot_water${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm_static

#-----------------------------------------------------------------

## Run: Hydration Water Self
export NAMD_ENERGY_LABEL="Hydration Water Self-interaction Energy"
export NAMD_ENERGY_SELECTION1="$selection_water_hydration"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_UPDATE_SELECTION1=1
export NAMD_ENERGY_UPDATE_SELECTION2=0
export NAMD_ENERGY_OUT_PREFIX="water_hydration_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm

## Run: Bulk Water Self
export NAMD_ENERGY_LABEL="Bulk Water Self-interaction Energy"
export NAMD_ENERGY_SELECTION1="$selection_water_bulk"
export NAMD_ENERGY_SELECTION2=""
export NAMD_ENERGY_UPDATE_SELECTION1=1
export NAMD_ENERGY_UPDATE_SELECTION2=0
export NAMD_ENERGY_OUT_PREFIX="water_bulk_self${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm

## Run: Protein-Hydration Water
export NAMD_ENERGY_LABEL="Protein Hydration Water Cross-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2="$selection_water_hydration"
export NAMD_ENERGY_UPDATE_SELECTION1=0
export NAMD_ENERGY_UPDATE_SELECTION2=1
export NAMD_ENERGY_OUT_PREFIX="prot_water_hydration${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-all"
run_openmm

## Run: Protein-Bulk Water
export NAMD_ENERGY_LABEL="Protein Bulk Water Cross-Interaction Energy"
export NAMD_ENERGY_SELECTION1="protein"
export NAMD_ENERGY_SELECTION2="$selection_water_bulk"
export NAMD_ENERGY_UPDATE_SELECTION1=0
export NAMD_ENERGY_UPDATE_SELECTION2=1
export NAMD_ENERGY_OUT_PREFIX="prot_water_bulk${OUT_FILE_SUFFIX}"
export NAMD_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm
