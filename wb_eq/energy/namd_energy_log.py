#!/usr/bin/env python3

import os
import re
import datetime
from collections.abc import Iterable
from itertools import repeat

import pandas as pd
import matplotlib.pyplot as plt
import time

"""
==================================================
Script to extract ENERGIES from NAMD .log file(s)
==================================================
It can also calculate energy stats (MIN, MAX, AVG) and Plot energies

Energies output by NAMD:
TS, BOND, ANGLE, DIHED, IMPRP, ELECT, VDW, BOUNDARY, MISC, KINETIC, TOTAL, TEMP, POTENTIAL, TOTAL3, TEMPAVG, PRESSURE, GPRESSURE, VOLUME, PRESSAVG, GPRESSAVG

NOTE: PRESSURE and VOLUME terms are only recorded for Peroidic systems, if outputPressure is set.
NOTE: "TS" stands for Time Step

USAGE:
1. copy this script to working dir
2. edit __name__ == "__main__" section of this script
2. run with "python namd_energy_log.py"
"""


def _extract_energies_internal(namd_log_files: [str, Iterable],
                               e_titles_callback: callable,
                               energies_callback: callable,
                               start_timestep: int = -1,
                               end_timestep: int = -1):
    """
    Base method to Extract energies from NAMD .log files and recive Callbacks

    @param namd_log_files : NAMD .log file or a list of log files
    @param e_titles_callback : e_titles_callback(energy_titles: list[str]) -> boolean
                            -> function that is called with ENERGY_TITLES (as list of strings) that are present in the first log file
                            Called only ONCE at the start:
                            -> returns a boolean. If True, continue reading the files, else stop and return

    @param energies_callback : a function that is called for each entry of all energies (in all log files)
                            It is guaranteed that "e_titles_callback(energy_titles)" will be called before
                            the first call to this function.
                            Energies are passed as list of strings, where each energy corresponds to the ENERGY_TITLE
                            of energy_titles list passed to e_titles_callback.
                            > energies_callback(energies: list[str])

    @param start_timestep : timestep to start reading energy_values from .log file(s).
                            -1 to start from the begining (any timestep present first)

    @param end_timestep : timestep to end reading energy_values from .log file(s).
                            -1 to read till the end of all .log files

    """

    if isinstance(namd_log_files, str):
        namd_log_files = [namd_log_files]

    e_titles = None
    has_range = start_timestep >= 0 or end_timestep >= 0

    energy_line_count = 0

    for log_file_path in namd_log_files:
        with open(log_file_path, 'r') as f:

            for line in f:
                line = line.strip()
                if e_titles is None:
                    if (line.startswith("ETITLE:") and (words := line.split())) and words[0] == "ETITLE:":
                        e_titles = words[1:]  # First will be Timestep
                        val = e_titles_callback(e_titles)
                        if not val:
                            return

                    continue

                # Now we have e_titles
                if (line.startswith("ENERGY:") and (words := line.split())) and words[0] == "ENERGY:":
                    energies = words[1:]  # First will be Timestep

                    if has_range:
                        ts = int(energies[0])
                        if (start_timestep >= 0 and ts < start_timestep) or (end_timestep >= 0 and ts > end_timestep):
                            continue

                    # Progress
                    energy_line_count += 1
                    if (energy_line_count % 100000) == 0:
                        print(f" Processing Line: {energy_line_count}", end='\r')

                    energies_callback(energies)


def _parse_energy_cols(energy_cols: [str, Iterable, None]) -> [list, None]:
    if isinstance(energy_cols, str):
        energy_cols = energy_cols.strip()
        if energy_cols:
            return energy_cols.split()
        return None

    if isinstance(energy_cols, (list, tuple, Iterable)):
        energy_cols = [e.strip() for e in energy_cols if isinstance(e, str) and e.strip()]
        if len(energy_cols) > 0:
            return energy_cols
        return None

    return None


def _out_filename_from_energy_cols(energy_cols: [list | None],
                                   suffix: str,
                                   default_file_name: str) -> str:
    if energy_cols and len(energy_cols) > 0:
        if len(energy_cols) == 1:
            return f"{energy_cols[0].lower()}{suffix}"

        return f"{energy_cols[0].lower()}-{energy_cols[-1].lower()}{suffix}"

    return default_file_name  # Default


def extract_energies(namd_log_files: [str, Iterable],
                     start_timestep: int = -1,
                     end_timestep: int = -1,
                     energy_cols: [str, Iterable, None] = None,
                     calc_stats: bool = True,
                     out_file_name: str = None,
                     out_stats_file_name: str = None,
                     out_delimiter: str = " \t ",
                     comment_token: str = "#",
                     comment_e_titles: bool = False):
    """
    Extract specified (or all) ENERGY values from NAMD .log file(s)
    within certain timestep range or for all time steps

    Generally available ENERGY_COLUMNS:
    BOND, ANGLE, DIHED, IMPRP, ELECT, VDW, BOUNDARY, MISC, KINETIC, TOTAL, TEMP, POTENTIAL, TOTAL3, TEMPAVG

    @param namd_log_files : NAMD .log file or a list of log files
    @param start_timestep : timestep to start reading energy_values from .log file(s).
                            -1 to start from the begining (any timestep present first)
    @param end_timestep : timestep to end reading energy_values from .log file(s).
                            -1 to read till the end of all .log files
    @param energy_cols : Energy fields required.
                        can be a single string, or list of strings, or None for all energy fields present in .log file(s)
    @param calc_stats : Whether to calculate energy stats: MIN, MAX and AVG (SLOWER)
    @param out_file_name : name of the output file. None for default
    @param out_stats_file_name : name of the output energy stats file. None for default
    @param out_delimiter : delimiter to use for output
    @param comment_token : A token for comments. Empty string to disable comments
    @param comment_e_titles : Whether to comment Energy Titles as well. Useful for programs
                            such as Xmgrace which cannot parse headers
    """

    energy_cols = _parse_energy_cols(energy_cols)
    if not out_file_name:
        out_file_name = _out_filename_from_energy_cols(energy_cols, suffix="_vs_ts.csv",
                                                       default_file_name="energies_vs_ts.csv")

    if calc_stats and not out_stats_file_name:
        out_stats_file_name = _out_filename_from_energy_cols(energy_cols, suffix=".stats.csv",
                                                             default_file_name="energies.stats.csv")

    print("==============================")
    print(f"-> EXTRACTING ENERGIES: [{', '.join(energy_cols) if energy_cols is not None else 'ALL'}]")
    print(f"-> Energy Stats: {'ON' if calc_stats else 'OFF'}")
    print('-------------------------------')

    indices = []
    titles = []  # energy titles: list[str]
    e_stats = []  # energy values min, max and avg: [[min_e1,min_e2,...], [max_e1,max_e2,...], [avg_e1,avg_e2,...]]
    e_count = [0]  # Hack to make it mutable

    with open(out_file_name, "w") as out_fd:
        def _erg_titles_callback(_e_titles: list[str]) -> bool:
            indices.clear()

            if energy_cols:
                if "TS" not in energy_cols and "TS" in _e_titles:
                    indices.append(_e_titles.index("TS"))
                indices.extend((_e_titles.index(c) for c in energy_cols if c in _e_titles))
            else:
                indices.extend(range(0, len(_e_titles)))  # All energy types

            if len(indices) == 0:
                print(" -> No Energy type found !!")
                return False

            # indices.sort()
            titles.clear()
            titles.extend((_e_titles[i] for i in indices))
            e_stats.clear()
            e_stats.extend((list(repeat(0, len(indices))) for _ in
                            range(3)))  # 3 rows, storing min, max and avg for each energy_type
            e_count[0] = 0

            # Comments
            if comment_token:
                comments = [
                    f"{comment_token}=================== NAMD ENERGY EXTRACTOR ==================",
                    f"{comment_token} INPUT NAMD .log file(s): [ {', '.join(namd_log_files)} ]",
                    f"{comment_token} INPUT Timestep range: [{start_timestep}, {end_timestep}]",
                    f"{comment_token} META File written on: {datetime.datetime.now()}",
                    f"{comment_token}-----------------------------"
                ]

                out_fd.write("\n".join(comments) + "\n")

            # Header: Energy Titles
            out_fd.write((comment_token if comment_token and comment_e_titles else "")
                         + out_delimiter.join(titles))

            return True

        def _erg_callback(energies: list[str]):
            out_fd.write("\n" + out_delimiter.join((energies[i] for i in indices)))

        def _erg_stats_callback(energies: list[str]):
            _line = ""

            for _i, ei in enumerate(indices):
                _line += energies[ei]
                if _i != len(indices) - 1:
                    _line += out_delimiter

                float_val = float(energies[ei])
                # First Call, set start min and max
                if e_count[0] == 0:
                    e_stats[0][_i] = float_val
                    e_stats[1][_i] = float_val
                else:
                    if (float_val <= e_stats[0][_i]):  # min
                        e_stats[0][_i] = float_val
                    elif (float_val > e_stats[1][_i]):  # max
                        e_stats[1][_i] = float_val

                e_stats[2][_i] += float_val  # sum (or avg)

            out_fd.write("\n" + _line)  # write output
            e_count[0] += 1

        _extract_energies_internal(namd_log_files=namd_log_files,
                                   start_timestep=start_timestep,
                                   end_timestep=end_timestep,
                                   e_titles_callback=_erg_titles_callback,
                                   energies_callback=_erg_stats_callback if calc_stats else _erg_callback)

        # calculate Stats
        stats_calculated = calc_stats and e_count[0] > 0
        if stats_calculated:
            for i in range(len(indices)):
                e_stats[2][i] /= e_count[0]  # calculate Average

            # Console Output
            print("\n------------------------")
            print(" AVERAGE ENERGIES:")
            print("\n".join(f"-> {titles[i]} : {e_stats[2][i]}" for i in range(len(indices))))

            # Writing Stats to File
            with open(out_stats_file_name, "w") as out_fd:

                # Comments
                if comment_token:
                    comments = [
                        f"{comment_token}=================== NAMD ENERGY STATS ==================",
                        f"{comment_token} INPUT NAMD .log file(s): [ {', '.join(namd_log_files)} ]",
                        f"{comment_token} INPUT Timestep range: [{start_timestep}, {end_timestep}]",
                        f"{comment_token} OUTPUT Total energy entries count: {e_count[0]}",
                        f"{comment_token} META File written on: {datetime.datetime.now()}",
                        f"{comment_token}-----------------------------"
                    ]

                    out_fd.write("\n".join(comments) + "\n")

                # Header: Energy Titles
                out_fd.write((comment_token if comment_token and comment_e_titles else "") + "   " + out_delimiter
                             + out_delimiter.join(titles))

                out_fd.write("\nmin" + out_delimiter + out_delimiter.join(map(str, e_stats[0])))
                out_fd.write("\nmax" + out_delimiter + out_delimiter.join(map(str, e_stats[1])))
                out_fd.write("\navg" + out_delimiter + out_delimiter.join(map(str, e_stats[2])))

        print("\n---------------------------------")
        print(f" OUTPUT Energy File: {out_file_name}")
        if stats_calculated:
            print(f" OUTPUT Stats File: {out_stats_file_name}")
        print(f" delimiter: '{out_delimiter}'")
        print(f" comment_token: '{comment_token}' | comment_energy_titles: {comment_e_titles}")
        print("===================================\n")


def find_files(folder_path: str,
               prefix: str = "",
               suffix: str = "",
               recursive: bool = False,
               sort: bool = True):
    """
    Finds all files in the specified folder with the given prefix and suffix,
    sorted so files with no number come first (both groups sorted naturally).

    :param folder_path: Path to the folder to search.
    :param prefix: Prefix string files should start with.
    :param suffix: Suffix string files should end with.
    :param recursive: If True, search subfolders recursively.
    :param sort: If True, sort files in natural order of their name
    :return: List of matching file paths, grouped and sorted as described.
    """
    matching_files = []
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith(prefix) and file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                if file.startswith(prefix) and file.endswith(suffix):
                    matching_files.append(full_path)

    if not sort:
        return matching_files

    # Sorting Naturally ---------------------
    def has_number(s):
        """Return True if the string contains any digits."""
        return any(char.isdigit() for char in os.path.basename(s))

    def natural_sort_key(s):
        """Split string into parts for natural sorting (numbers as ints)."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

    # Split into files without numbers and with numbers
    files_num = []  # files with number in their name
    files_no_num = []  # files with no number in their name
    for f in matching_files:
        if has_number(f):
            files_num.append(f)
        else:
            files_no_num.append(f)

    # Sort both groups naturally
    files_no_num.sort(key=natural_sort_key)
    files_num.sort(key=natural_sort_key)

    # Combine
    return files_no_num + files_num


if __name__ == '__main__':
    #### Energies output by NAMD:
    # ---------------------
    # Available Columns:
    # ---------------------
    # TS, BOND, ANGLE, DIHED, IMPRP, ELECT, VDW, BOUNDARY, MISC, KINETIC, TOTAL, TEMP, POTENTIAL, TOTAL3, TEMPAVG, PRESSURE, GPRESSURE, VOLUME, PRESSAVG, GPRESSAVG

    ## ====================  INPUT CONFIG  =========================
    # TODO: input .log file(s)
    namd_log_files = ["../amyl_wb_eq2.log"]
    # namd_log_files = find_files("..", prefix="amyl_wb_eq", suffix=".log", recursive=False, sort=True)

    # TODO: ENERGY COLUMNS ->  str (whitespace delimited) OR list[str] OR None (for all columns)
    energy_columns = None  # All columns
    # energy_columns = "TS BOND ANGLE DIHED IMPRP ELECT VDW KINETIC POTENTIAL TOTAL TEMP PRESSURE VOLUME"     # As string
    # energy_columns = ["BOND", "TEMP", "KINETIC"]      # As list

    timestep_start = -1  # TODO: START Timestep, or -1 to start form beginning
    timestep_end = -1  # TODO: END Timestep, or -1 to read till the end of all .log file(s)
    calculate_stats = True  # Calculate Energies MIN, MAX, AVG (SLOWER)

    ## =====================  OUTPUT CONFIG  ==========================
    out_energy_file = "energy-all.csv"  # File for energy data. None or empty str for default
    out_energy_stats_file = "energy-all.stats.csv"  # File for average energy data. None or empty str for default
    out_delimiter = " "
    comment_token = "#"  # Empty string to disable comments
    comment_e_titles = False  # whether to comment ENERGY TITLES in the Header, useful for Xmgrace

    ## ----------------------------------------------------------------

    start_time = time.time()

    ## EXTRACT ENERGIES vs TS
    do_extract_energies = True
    if do_extract_energies:
        extract_energies(namd_log_files=namd_log_files,
                         energy_cols=energy_columns,
                         start_timestep=timestep_start,
                         end_timestep=timestep_end,
                         calc_stats=calculate_stats,
                         out_file_name=out_energy_file,
                         out_stats_file_name=out_energy_stats_file,
                         out_delimiter=out_delimiter,
                         comment_token=comment_token,
                         comment_e_titles=comment_e_titles)

    ## PLOTTING
    # TODO: set comment_e_titles = False for this to work
    do_plot_energies = False
    if do_plot_energies:
        df = pd.read_csv(out_energy_file, sep=r"\s+", comment=comment_token)

        # Filter if needed
        # df = df[df["TS"] >= 100]

        # TODO: set labels accordingly
        # plt.plot(df["TS"], df["TOTAL"], label="Total")
        plt.plot(df["TS"], df["ELECT"], label="Elect")

        plt.legend(loc="best")
        plt.show()

    print(f"Time Taken: {time.time() - start_time} s")
