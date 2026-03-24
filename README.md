# simple-mpc

A small quadruped MPC sandbox based on `aligator`, `pinocchio`, and the Go2 model from `example-robot-data`.

## Setup

```bash
conda env create -f environment.yml
conda activate opti2amp
```

## Run

Ideal MPC demo:

```bash
python main.py
```

Batch recording:

```bash
python main_record.py
```

## Notes

- Core MPC code is in `mpc/`.
- Recording utilities are in `record/`.
- Recording settings are defined in `record/config.py`.
