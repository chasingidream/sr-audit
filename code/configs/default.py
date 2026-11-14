"""Default configuration for the spatiotemporal SR project.

Override any field by passing --cfg path/to/your.yaml (JSON/YAML) or editing here.
Runnable out-of-the-box on synthetic data (no downloads) for a smoke test.
"""
CONFIG = {
    # --- model ---
    "in_ch": 3,
    "feat": 64,
    "scale": 4,
    "T": 5,                      # number of low-res multi-temporal frames
    "lr_size": 64,               # canonical LR patch size (HR = lr_size*scale)
    "n_groups": 4,
    "n_blocks": 4,
    "rsa_radius": 4,
    "use_hr_ref": True,
    "use_conf": True,

    # --- data ---
    "dataset": {
        "type": "synthetic",     # synthetic | simulated | worldstrat
        "kwargs": {"n_samples": 2000, "scale": 4, "lr_size": 64, "T": 5, "seed": 0},
        # "type": "worldstrat", "root": "/path/to/worldstrat/data"
    },

    # --- training ---
    "stage": "end2end",          # baseline | rsa | end2end | full
    "batch_size": 8,
    "epochs": 50,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_every": 5,
    "resume": None,

    # --- losses ---
    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        # perceptual (LPIPS) is lazy; set >0 only after the alexnet/vgg weights
        # are cached once (slow download on this network). Default off for runs.
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 1.0,
        "w_cyclic": 0.1,
    },

    # --- io ---
    "save_dir": "experiments/ckpts",
    "log_dir": "experiments/logs",
    "project_root": "..",        # relative to code/ ; resolved in train.py
}
