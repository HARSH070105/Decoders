# Decoders

Quantum error-correction decoder experiments for BP, BP+OSD, and a Tesseract-style search prototype.

The repository bundles parity-check matrices for several code families and a few standalone simulation scripts that compare decoder performance under depolarizing noise.

## Requirements

Install the Python packages used by the scripts:

```bash
pip install numpy scipy matplotlib tqdm
```

## How To Run

Run the scripts from the workspace root, `c:\Users\harsh\OneDrive\Desktop\QECC`, so the `Decoders...` imports resolve correctly.

Examples:

```bash
python Decoders\BP_OSD_Decoder\decoder.py
python Decoders\BP_OSD_Decoder\New_trial.py
python Decoders\Tesseract_Decoder\Decoder.py
```

The `.npz` files in the code-family folders contain the parity-check matrices loaded by the scripts:

- `gb_254_28_hx.npz`, `gb_254_28_hz.npz`
- `ghgp_882_24_hx.npz`, `ghgp_882_24_hz.npz`
- `hgp_7938_578_hx.npz`, `hgp_7938_578_hz.npz`
