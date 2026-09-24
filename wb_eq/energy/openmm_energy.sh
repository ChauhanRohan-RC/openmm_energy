#!/bin/bash

# ======================================================================
# Launcher script for openmm_energy.py
# ======================================================================
# OpenMM and MDAnalysis implementation of NAMD PairInteraction Energy
# Optimized for GPUs
# ------------------------------------
# TODO: set env vars for each run


# Exit on error
#set -e

# for cpptraj chunking support
#module load ambertools


run_openmm() {
	python3 openmm_energy.py
}

run_openmm_static() {
	export OPENMM_ENERGY_UPDATE_SELECTION1=0
	export OPENMM_ENERGY_UPDATE_SELECTION2=0
	run_openmm
	unset OPENMM_ENERGY_UPDATE_SELECTION1
	unset OPENMM_ENERGY_UPDATE_SELECTION2
}


# 2nd HYDRATION SHELL CUTOFF (from RDF vs r plot minima)
# TODO: use mdanalysis selection syntax
export selection_water_hydration="water and around 4.25 protein"
export selection_water_bulk="water and not around 5.0 protein"

# TEST
OUT_FILE_SUFFIX="7"


# Run 1: Protein Self
export OPENMM_ENERGY_LABEL="Protein Self-Interaction Energy"
export OPENMM_ENERGY_SELECTION1="protein"
export OPENMM_ENERGY_SELECTION2=""
export OPENMM_ENERGY_OUT_PREFIX="prot_self${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-conf -nonb -pote"
run_openmm_static

## Run 2: Water Self
export OPENMM_ENERGY_LABEL="Water Self-Interaction Energy"
export OPENMM_ENERGY_SELECTION1="water"
export OPENMM_ENERGY_SELECTION2=""
export OPENMM_ENERGY_OUT_PREFIX="water_self${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm_static

## Run 3: Protein-Water
export OPENMM_ENERGY_LABEL="Protein-Water Cross-Interaction Energy"
export OPENMM_ENERGY_SELECTION1="protein"
export OPENMM_ENERGY_SELECTION2="water"
export OPENMM_ENERGY_OUT_PREFIX="prot_water${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm_static

#-----------------------------------------------------------------

## Run: Hydration Water Self
export OPENMM_ENERGY_LABEL="Hydration Water Self-interaction Energy"
export OPENMM_ENERGY_SELECTION1="$selection_water_hydration"
export OPENMM_ENERGY_SELECTION2=""
export OPENMM_ENERGY_UPDATE_SELECTION1=1
export OPENMM_ENERGY_UPDATE_SELECTION2=0
export OPENMM_ENERGY_OUT_PREFIX="water_hydration_self${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm

## Run: Bulk Water Self
export OPENMM_ENERGY_LABEL="Bulk Water Self-interaction Energy"
export OPENMM_ENERGY_SELECTION1="$selection_water_bulk"
export OPENMM_ENERGY_SELECTION2=""
export OPENMM_ENERGY_UPDATE_SELECTION1=1
export OPENMM_ENERGY_UPDATE_SELECTION2=0
export OPENMM_ENERGY_OUT_PREFIX="water_bulk_self${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm

## Run: Protein-Hydration Water
export OPENMM_ENERGY_LABEL="Protein Hydration Water Cross-Interaction Energy"
export OPENMM_ENERGY_SELECTION1="protein"
export OPENMM_ENERGY_SELECTION2="$selection_water_hydration"
export OPENMM_ENERGY_UPDATE_SELECTION1=0
export OPENMM_ENERGY_UPDATE_SELECTION2=1
export OPENMM_ENERGY_OUT_PREFIX="prot_water_hydration${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-all"
run_openmm

## Run: Protein-Bulk Water
export OPENMM_ENERGY_LABEL="Protein Bulk Water Cross-Interaction Energy"
export OPENMM_ENERGY_SELECTION1="protein"
export OPENMM_ENERGY_SELECTION2="$selection_water_bulk"
export OPENMM_ENERGY_UPDATE_SELECTION1=0
export OPENMM_ENERGY_UPDATE_SELECTION2=1
export OPENMM_ENERGY_OUT_PREFIX="prot_water_bulk${OUT_FILE_SUFFIX}"
export OPENMM_ENERGY_OUT_ENERGIES="-nonb -pote"
run_openmm
