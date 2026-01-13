# Hyperparameters

# game
BOARD_WIDTH = 15
COUNT_LEN = 5

# network
DN_FILTERS = 256
DN_KERNEL_SIZE = 3
DN_BLOCK_NUM = 12
PREV_STATE_COUNT = 3

# record preprocessing
RIF_DATABASE_PATH = 'training_data/renjunet_v10_20260105.rif'
XML_DATABASE_PATH = 'training_data/games.xml'
MAX_DATA_LENGTH = 150000
GAME_LENGTH_THRES = 50

# supervised_learning
SL_DATASET_DIR = "./training_data/dataset"
SL_CHUNK_SIZE = 40000
SL_CHECKPOINT_FILE = "train_supervised_checkpoint.json"
SL_MODEL_PATH = "./model/best_supervised.pth"
SL_EPOCHS = 50
SL_LEARNING_RATE = 0.001
SL_SCHEDULER_GAMMA = 0.95
SL_SCHEDULER_STEP_SIZE = 50
SL_BATCH_SIZE = 512
SL_TEST_RATIO = 0.1
SL_TEST_INTERVAL = 50

# self_play
RL_CHECKPOINT_FILE = "train_sp_state.json"
RL_GAME_CNT = 240
RL_TEMP = 0.1
RL_CORES = 12
RL_MCTS_CNT = 400
RL_EPOCHS = 10
RL_LEARNING_RATE = 0.002
RL_BATCH_SIZE = 256

# evaluate_network
EN_GAME_COUNT = 240  # 평가 1회 당 게임 수 (오리지널: 400)
EN_TEMPERATURE = 0.1  # 볼츠만 분포 온도
EN_AVERAGE_POINT = 0.5
EN_NUM_CORES = 12
EN_MCTS_COUNT = 400

# play
PLAY_TEMPERATURE = 0.0
PLAY_MCTS_COUNT = 600

# wandb
USE_WANDB = True
WANDB_PROJECT = "gomokupp"