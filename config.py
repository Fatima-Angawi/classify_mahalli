import os
SEED        = 42

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
TEST_SIZE         = 0.15
VAL_SIZE          = 0.15
RANDOM_STATE      = 42
MAX_LEN     = 256        # product descriptions are usually short
BATCH_SIZE  = 16
EPOCHS      = 6
LR          = 2e-5
OUTPUT_DIR  = "./salla_scam_model"
DATA_PATH   = "products.csv"   # <-- your CSV: columns ['text', 'label']
                               #     label: 0=legit, 1=scam
