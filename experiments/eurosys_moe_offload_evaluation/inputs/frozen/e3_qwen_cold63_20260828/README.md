# E3 Qwen cold63 frozen inputs

These files are versioned inputs, not experiment outputs. They were derived
with `freeze_qwen_e3_plan.py` using 63 cold experts per layer, calibration batch
32, and 224 expected decode tokens.

- routing source SHA256: `912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81`
- calibration fixture SHA256: `468d5cf35f231de18978cd7ba3873ebcb60cac9d15dc7486f47796f35dc4dae7`
- `exact_parameter_names.txt` SHA256: `6142b29ba13671bd2c0c1e18525a64e1bd3f676ad369add9d3d2323086ac4141`
- `cold_decode_plan_2g.json` SHA256: `05686bd2c4df0f5c052c90902e5945bee9d921a8d36b7b1009712dd314cc8620`
