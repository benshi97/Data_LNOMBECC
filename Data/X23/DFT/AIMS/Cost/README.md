# X23 hybrid-DFT composite cost archive

This directory contains the 138 FHI-aims outputs needed to measure the total
cost of the three-calculation hybrid-DFT composite:

`E(composite) = E[hybrid/lightdense] + E[GGA/tight] - E[GGA/lightdense]`

The computational cost is additive, so all three calculations are counted for
both the periodic crystal and isolated molecule. Systems are in the PRL/DMC
order used by the existing `Data/X23/DFT/AIMS/{01..23}` archive.

| File | Calculation |
|---|---|
| `aims_01.out.gz` | B86bPBE-50 + XDM (0.74, 1.72), lightdense |
| `aims_02.out.gz` | B86bPBE + default XDM, lightdense |
| `aims_03.out.gz` | B86bPBE + default XDM, tight |

The periodic files are the new Popeye runs. The method-1 molecular file is the
matching archived B86bPBE-50/lightdense calculation with default XDM damping,
used only as a timing proxy. XDM damping is evaluated post-SCF, so changing
`a1/a2` does not change the costly hybrid SCF procedure; its XDM evaluation is
negligible compared with that SCF. Methods 2 and 3 use their matching archived
molecular calculations directly.

`manifest.json` records every source, timing, MPI-rank count, uncompressed hash,
and whether an entry is a timing proxy. Rebuild from the LNOMBECC project root:

```bash
conda run -n skzcam python \
  Analysis/26_09_17-X23_Hybrid_DFT_Cost/stage_cost_outputs.py
```

The cost analyses are included in `analyse.ipynb`, under **Overall comparison
to periodic HF and DMC** and **Computational cost for X23 dataset**. They use
128 MPI ranks for the molecular calculations and 64 for the periodic crystals,
as recorded in the outputs and manifest.
