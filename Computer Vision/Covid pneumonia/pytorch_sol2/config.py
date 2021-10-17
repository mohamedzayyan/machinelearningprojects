TRAIN_DIR = "./datasets/images_training_rev1/images/"
TRAIN_CSV = "./classes/gz2_hub.csv"

TEST_DIR ="./datasets/edit/"
TEST_CSV = "./classes/usethis_rsa2.csv"

EFIGI_DIR ="./datasets/efigi/pics/png/"
EFIGI_CSV = "./classes/efigi_model.csv"

BATCH_SIZE = 40
DEVICE = "cuda" 
VALIDATION_SPLIT = 0.05
RANDOM_SEED = 42
SHUFFLE_DS = True

CSV_HEADER = ['objid', 'sample', 'asset_id', 'dr7objid',
       'hubble_type', 'E0', 'E3-5', 'E7', 'Irr', 'S0', 'SBa', 'SBb', 'SBc', 'Sa', 'Sb', 'Sc']

CSV_HEADER_rsa = ['Galaxy_ID','Type', 'E0', 'E3-5', 'E7', 'Irr', 'S0', 'SBa', 'SBb', 'SBc', 'Sa', 'Sb', 'Sc']
CSV_HEADER_efigi = ['pgc_name','full_pgc_name', 'hubb', 'E', 'Irr', 'S0', 'SBa', 'SBb', 'SBc', 'Sa', 'Sb', 'Sc']


