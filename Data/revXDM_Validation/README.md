# revXDM validation outputs

S66 and CP1b contain `{system}/aims_{revXDM,b86bpbe,pbed3}.out.gz`.
G60 contains `{system}/{method}/{crystal,molecule}/` with one gzip output per
folder: `aims_revXDM.out.gz` for `01-b86bpbe50-xdm074-lightdense`,
`aims_b86bpbe.out.gz` for `03-b86bpbe-xdmdefault-tight`, and
`aims_pbed3.out.gz` for `04-pbe-d3bj-tight`. B86bPBE-50 with default XDM
(`02-...`) is not included.
Each G60 file is directly gzip-compressed, with no tar layer. The source
calculations remain unchanged on Popeye. `methods.npy` maps the S66/CP1b output
indices to their functional, dispersion parameters, and basis.

- `S66/final_ccsdt_references.npy`: 66 unrounded final CCSD(T) interaction energies
  from the FNDMC-S66 analysis, now used in `analyse.ipynb`. `reference_values`
  contains kJ/mol values and `formatted_names` contains the matching table labels,
  both keyed by `S66_1` through `S66_66`. The average uses `Hobza_2`, `Martin_Gold`
  (or `Martin_Silver` when Gold is missing), and `14k-Gold` from `Hobza_Nagy.csv`.
  Original kcal/mol averages, the three components, population standard deviations,
  source checksums, and the existing geometry mapping are retained. Conversion to
  kJ/mol uses exactly 4.184; negative values indicate binding.
- `S66/references.npy`: the previous 66 GSCDB CCSD(T) interaction-energy references,
  retained unchanged for comparison. The
  `reference_values` dictionary is in kJ/mol, with negative values for binding.
  `stoichiometry` identifies the dimer and two monomer folders for each reaction.
  The original hartree references are also retained.
- `CP1b/references.npy`: 54 DLPNO-CCSD(T0)/CBS conformational-energy references
  in the `reference_values` dictionary, in kJ/mol. `groups` maps the 20 species
  to their conformers, and `reference_conformers` identifies each zero anchor.
  Use the same anchor for all DFT methods; the standard MAD excludes the 20
  zero-reference conformers and includes the remaining 34 differences.
- `G60/g60_experimental_lattice_energies_kj_mol.npy`: experimental lattice
  energies in kJ/mol, keyed by the 60 G60 system names. The G60 validation in
  `analyse.ipynb` reads this local copy.

Load any dictionary with `np.load(path, allow_pickle=True).item()`.
The dictionary structure and kJ/mol units are unchanged. All archived
dictionaries use `.npy`; superseded JSON versions have been removed.

`output_manifest.npy` records each output's source path, uncompressed SHA-256,
file sizes, and verified final dispersion-inclusive energy. All gzip files
were checked against the remote source and the MAD analysis energy snapshot.

The reusable copy/verification script is
`Analysis/26_09_14-S66_CP1b_MAD/archive_outputs.py` relative to the project root;
run it in the `skzcam` environment. A rerun preserves matching archived files
and refuses to replace different data.

The G60 gzip checksums and source paths are recorded in
`Analysis/26_09_21-G60_Output_Archive/g60_gzip_manifest.json` relative to the
project root. All 360 gzip files were verified against the original Popeye
`aims.out` checksums.

Regenerate the final S66 reference dictionary with
`conda run -n skzcam python Analysis/26_09_14-S66_CCSDT_References/save_final_ccsdt_references.py`
from the project root. The script reads `/Users/bshi/Projects/DMC/FNDMC-S66`
without changing it, verifies the 66 system/geometry identities, and preserves
matching output rather than overwriting different data.
