#!/usr/bin/env python
# coding: utf-8

# Description: Contains functions for reading and plotting data from MRCC, ORCA and VASP calculations.

import numpy as np
from datetime import datetime
import pandas as pd
import re
import pyblock
import gzip


from ase.db import connect
from ase.units import kcal, kJ, mol, Hartree
from ase.io import read,write
from ase import Atoms
import os

kcalmol = kcal*1000/mol
kjmol = kJ/mol*1000


# Define units
kB = 8.617330337217213e-05
mol = 6.022140857e+23
kcal = 2.611447418269555e+22
kJ = 6.241509125883258e+21
Hartree = 27.211386024367243
Bohr = 0.5291772105638411


# Some basic conversion factors
cm1_to_eV = 1 / 8065.54429
hundredcm1 = 100 * cm1_to_eV * 1000
kcalmol_to_meV = kcal / mol * 1000
kjmol_to_meV = kJ / mol * 1000
mha_to_meV = Hartree


def reverse_search_for(lines_obj, keys, line_start=0):
    for ll, line in enumerate(lines_obj[line_start:][::-1]):
        if any([key in line for key in keys]):
            return len(lines_obj) - ll - 1


# Read Total time                                  : from output file
def read_total_time_from_aims_output_file(rundir):
    with open(rundir) as f:
        lines = f.readlines()

        line_start = reverse_search_for(lines, ["Total time                                  :"])

        time = float(lines[line_start].split()[-2])
        
    return time


# Function to calculate CCSD(cT) from CCSD(T):

def cT_calc(mp2_corr,ccsd_corr,ccsdt_corr):
    t_corr = ccsdt_corr - ccsd_corr
    X = 0.7764+0.278*(mp2_corr/ccsd_corr)
    ct_corr = t_corr / X
    ccsdct_corr = ccsd_corr + ct_corr
    return ccsdct_corr


def get_enthalpy_classical_md (kinetic_energy, potential_energy,
                               volume, number_of_molecules, pressure_bar=1):

    # compute classical enthalpy at each step
    
    pressure_GPa = pressure_bar / 10**4
    pV = pressure_GPa * volume / 160.2176 # in eV
    
    classical_enthalpy = kinetic_energy + potential_energy + pV # in eV
    
    # compute average classical enthalpy 
    
    av_classical_enthalpy = np.mean(classical_enthalpy)/ number_of_molecules

    
    # compute error with reblocking
    
    reblock_data = pyblock.blocking.reblock(classical_enthalpy)
    opt = pyblock.blocking.find_optimal_block(len(classical_enthalpy), reblock_data)
    error_with_reblock = float(reblock_data[opt[0]].std_err)
    
    error_classical_enthalpy = error_with_reblock/number_of_molecules
    
    
    return av_classical_enthalpy, error_classical_enthalpy # in eV/mol


def get_qe_walltime(filename):

    def parse_time_to_hours(time_str):
        # Regular expression to capture days, hours, minutes, seconds (including decimals)
        pattern = r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?'
        match = re.fullmatch(pattern, time_str.strip())
        
        if not match:
            raise ValueError(f"Invalid time format: {time_str}")
        
        days, hours, minutes, seconds = match.groups()
        total_hours = 0.0
        if days:
            total_hours += int(days) * 24
        if hours:
            total_hours += int(hours)
        if minutes:
            total_hours += int(minutes) / 60
        if seconds:
            total_hours += float(seconds) / 3600
        
        return total_hours

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Check if 'PWSCF' is in lines[-8].split()[0]
    if 'PWSCF' not in lines[-8].split()[0]:
        raise ValueError(f"Unexpected format in QE output file: {filename}")
    
    # The time is in between the 'CPU' and 'Wall' in e.g., "    PWSCF        :  47m51.91s CPU  51m 0.93s WALL"
    timing_line = lines[-8].split()
    # Find idx of 'CPU' 
    cpu_idx = [idx for idx, val in enumerate(timing_line) if val == 'CPU'][0]
    timing_str = ''.join(lines[-8].split()[cpu_idx+1:-1])
        
    return parse_time_to_hours(timing_str)



# Getting the cost of the DFT calculations
def get_vasp_walltime(filename):
    """
    Reads the walltime from the OUTCAR file (plain or .gz).

    Parameters
    ----------
    filename : str
        The location of the 'OUTCAR' file to read from.

    Returns
    -------
    float
        The walltime in seconds.
    """

    open_func = gzip.open if filename.endswith(".gz") else open

    with open_func(filename, "rt") as f:
        for line in f:
            if "Elapsed time (sec):" in line:
                return float(line.split()[-1])

    raise ValueError("Elapsed time not found in file")



def get_vasp_energy(filename):
    """
    Parse the final VASP energy (without entropy) from an OUTCAR file
    or its gzip-compressed version.

    Parameters
    ----------
    filename : str
        Path to the OUTCAR file (.gz/.gzip supported).

    Returns
    -------
    float
        The energy in the original units.

    Raises
    ------
    ValueError
        If the energy line is not found in the file.
    """

    search_word = "energy  without entropy="

    open_func = gzip.open if filename.endswith((".gz", ".gzip")) else open

    with open_func(filename, "rt", encoding="ISO-8859-1") as fp:
        for line in fp:
            if search_word in line:
                return float(line.split()[-1])

    raise ValueError(f"VASP energy not found in file: {filename}")

    

def convert_df_to_latex_input(
    df,
    start_input = '\\begin{table}\n',
    end_input = '\n\\end{table}',
    label = "tab:default",
    caption = "This is a table",
    replace_input = {},
    df_latex_skip = 0,
    adjustbox = 0,
    scalebox = False,
    multiindex_sep = "",
    filename = "./table.tex",
    index = True,
    column_format = None,
    center = False,
    rotate_column_header = False,
    output_str = False,
    float_fmt = None
):
    if column_format is None:
        column_format = "l" + "r" * len(df.columns)
    
    if label != "":
        label_input = r"\label{" + label + r"}"
    else:
        label_input = ""
    caption_input = r"\caption{" + label_input + caption +  "}"

    if rotate_column_header:
        df.columns = [r'\rotatebox{90}{' + col + '}' for col in df.columns]

    with pd.option_context("max_colwidth", 1000):
        if float_fmt is not None:
            df_latex_input = df.to_latex(escape=False, column_format=column_format,multicolumn_format='c', multicolumn=True,index=index,float_format=float_fmt)
        else:
            df_latex_input = df.to_latex(escape=False, column_format=column_format,multicolumn_format='c', multicolumn=True,index=index)
    for key in replace_input:
        df_latex_input = df_latex_input.replace(key, replace_input[key])
    
    df_latex_input_lines = df_latex_input.splitlines()[df_latex_skip:]
    # Get index of line with midrule
    toprule_index = [i for i, line in enumerate(df_latex_input_lines) if "toprule" in line][0]
    df_latex_input_lines[toprule_index+1] = df_latex_input_lines[toprule_index+1] + ' ' + multiindex_sep
    df_latex_input = '\n'.join(df_latex_input_lines)
    end_adjustbox = False

    if output_str:
        latex_string = ""
        latex_string += start_input + "\n"
        latex_string += caption_input + "\n"
        if center == True and adjustbox == 0:
            latex_string += r"\begin{adjustbox}{center}" + "\n"
            end_adjustbox = True
        elif adjustbox > 0 and center == False:
            latex_string += r"\begin{adjustbox}{max width=" + f"{adjustbox}" + r"\textwidth}" + "\n"
            end_adjustbox = True    
        elif adjustbox > 0 and center == True:
            latex_string += r"\begin{adjustbox}{center,max width=" + f"{adjustbox}" + r"\textwidth}" + "\n"
            end_adjustbox = True
        if scalebox:
            latex_string += r"\begin{adjustbox}{scale=" + f"{scalebox}" + "}" + "\n"
            end_adjustbox = True
        latex_string += df_latex_input
        if end_adjustbox:
            latex_string += "\n\\end{adjustbox}"
        latex_string += "\n" + end_input
        return latex_string

    else:
        with open(filename, "w") as f:
            f.write(start_input + "\n")
            f.write(caption_input + "\n")
            if center == True and adjustbox == 0:
                f.write(r"\begin{adjustbox}{center}" + "\n")
                end_adjustbox = True
            elif adjustbox > 0 and center == False:
                f.write(r"\begin{adjustbox}{max width=" + f"{adjustbox}" + r"\textwidth}" + "\n")
                end_adjustbox = True    
            elif adjustbox > 0 and center == True:
                f.write(r"\begin{adjustbox}{center,max width=" + f"{adjustbox}" + r"\textwidth}" + "\n")
                end_adjustbox = True
            if scalebox:
                f.write(r"\begin{adjustbox}{scale=" + f"{scalebox}" + "}" + "\n")
                end_adjustbox = True
            f.write(df_latex_input)
            if end_adjustbox:
                f.write("\n\\end{adjustbox}")
            f.write("\n" + end_input)
        
