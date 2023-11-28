IMAGE_REGISTRY = "hsteude"


SIM_SUBSYSTEM_MAP = {
    "a": [
        "sig_a1",
        "sig_a2",
        "sig_a3",
    ],
    "b": [
        "sig_b1",
        "sig_b2",
        "sig_b3",
    ],
}

SIM_INPUT_COLS = [
    "sig_a1",
    "sig_a2",
    "sig_a3",
    "sig_b1",
    "sig_b2",
    "sig_b3",
]

SIM_SUBSYSTEM_MAP_COMP_A = {
    "a": [
        "sig_a1",
        "sig_a2",
        "sig_a3",
    ],
}
SIM_INPUT_COLS_COMP_A = [
    "sig_a1",
    "sig_a2",
    "sig_a3",
]
SIM_SUBSYSTEM_MAP_COMP_B = {
    "b": [
        "sig_b1",
        "sig_b2",
        "sig_b3",
    ],
}
SIM_INPUT_COLS_COMP_B = [
    "sig_b1",
    "sig_b2",
    "sig_b3",
]


SIM_SEQ_LEN = 1000

SWAT_SENSOR_COLS = [
    "FIT101",
    "LIT101",
    "MV101",
    "P101",
    "P102",
    "AIT201",
    "AIT202",
    "AIT203",
    "FIT201",
    "MV201",
    "P201",
    "P202",
    "P203",
    "P204",
    "P205",
    "P206",
    "DPIT301",
    "FIT301",
    "LIT301",
    "MV301",
    "MV302",
    "MV303",
    "MV304",
    "P301",
    "P302",
    "AIT401",
    "AIT402",
    "FIT401",
    "LIT401",
    "P401",
    "P402",
    "P403",
    "P404",
    "UV401",
    "AIT501",
    "AIT502",
    "AIT503",
    "AIT504",
    "FIT501",
    "FIT502",
    "FIT503",
    "FIT504",
    "P501",
    "P502",
    "PIT501",
    "PIT502",
    "PIT503",
    "FIT601",
    "P601",
    "P602",
    "P603",
]

SWAT_SYMBOLS_MAP = {
    "Comp_1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
    "Comp_2": [
        "AIT201",
        "AIT202",
        "AIT203",
        "FIT201",
        "MV201",
        "P201",
        "P202",
        "P203",
        "P204",
        "P205",
        "P206",
    ],
    "Comp_3": [
        "DPIT301",
        "FIT301",
        "LIT301",
        "MV301",
        "MV302",
        "MV303",
        "MV304",
        "P301",
        "P302",
    ],
    "Comp_4": [
        "AIT401",
        "AIT402",
        "FIT401",
        "LIT401",
        "P401",
        "P402",
        "P403",
        "P404",
        "UV401",
    ],
    "Comp_5": [
        "AIT501",
        "AIT502",
        "AIT503",
        "AIT504",
        "FIT501",
        "FIT502",
        "FIT503",
        "FIT504",
        "P501",
        "P502",
        "PIT501",
        "PIT502",
        "PIT503",
    ],
    "Comp_6": ["FIT601", "P601", "P602", "P603"],
}

SWAT_HPARAMS = dict(
    batch_size=64,
    learning_rate=0.0005,
    kernel_size=15,
    beta=1e-4,
)
