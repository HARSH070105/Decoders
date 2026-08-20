# Quantum Error-Correcting Codes' Decoders

Monte Carlo experiments for quantum low-density parity-check (QLDPC) codes. The repository compares belief propagation (BP) with BP followed by ordered-statistics decoding (BP-OSD) under several noise models. The BP-OSD algortihm is from [ldpc](https://github.com/quantumgizmos/ldpc) repository, by quantumqizmos.

## Repository Layout

```Shell
BP-OSD_Decoder/
├── Other Codes/       Generalized bicycle and hypergraph-product codes
│   ├── Codes/         Stored parity-check matrices (`.npz`)
│   └── Results_*/     Simulation output
├── Tile Codes/        Tile-code construction and simulations
│   ├── Codes/         Stored tile-code matrices and text descriptions
│   └── Results_*/     Simulation output
└── HaRT Codes/        Placeholder for HaRT-code experiments

Possibly_Wrong/        Experimental bias-tailored implementation maybe wrogn, needsa checking
```

The `Possibly_Wrong` directory is kept separately because its channel and decoder implementation is experimental and should not be treated as the reference results.

## Requirements

Use Python 3.9 or newer and install the packages used by the scripts:

```powershell
python -m pip install numpy scipy matplotlib tqdm ldpc
```

## Running Simulations

Run each script from its own directory. This is required because the scripts import sibling modules such as `Channel`, `Decoders`, and `Failure` by filename.

### Other codes

```powershell
cd "BP-OSD_Decoder\Other Codes"
python Simulate.py
```

Edit `code_name`, `error_rates`, and `trials` near the bottom of `Simulate.py` to select a code and simulation size. Available stored codes include:

- `gb_126_28`
- `gb_254_28`
- `ghp_882_24`

This script evaluates `depolarizing`, `pure_x`, and `pure_z` noise and writes one comparison file per model to `Results_<code_name>\`.

### Tile codes

```powershell
cd "BP-OSD_Decoder\Tile Codes"
python Simulate.py
```

The tile simulation is configured in the same way.  Available stored codes include `tile_288_8_12` and `tile_288_8_14`.

### Experimental bias-tailored code

From the repository root:

```powershell
python Possibly_Wrong\simulate_p_adjusted.py
```

This uses the code selected by `code_name` in the script and stores output in `Possibly_Wrong\results\`.

## Result Files and Plots

Simulation files are plain text with three columns:

```text
physical_error_rate  WER_BP  WER_BP-OSD
```

The plotting scripts read these files to produce word-error-rate (WER) comparisons on logarithmic axes. Existing result directories contain the checked-in data used for the comparison plots.

## Code Generation

Code-construction scripts are provided for rebuilding or modifying code matrices:

```powershell
cd "BP-OSD_Decoder\Other Codes"
python Build_codes.py

cd "..\Tile Codes"
python Build_Tile_Codes.py
```

Generated matrices are saved under the corresponding `Codes\` directory. Review the parameters in the builder before generating a new code.

## Notes

- `.npz` code files store sparse `HX` and `HZ` matrices.
- Simulations can be computationally expensive because each physical error rate runs many Monte Carlo trials; reduce `trials` for a quick smoke test.
- The scripts compute canonical logical operators and count a trial as a failure when the residual error contains a logical operator.
