
# Hyperparameters

# game
board_width = 15
count_len = 5

# mcts
pv_evaluate_cnt = 200

# generate_record_list
xml_path = 'training_data/games.xml'
max_record_cnt = 150000

# network
dn_filters = 128
dn_kernel_size = 3
dn_block_num = 8

# record_play
REC_START_IDX = 0
FROM_RECORD = False
FROM_SELF_PLAY = True

# self_play
sp_game_cnt = 240
sp_temperature = 0.1
sp_num_cores = 12

# train_network
rn_epochs = 20
batch_size = 32

# train_cycle
record_batch_size = 500
train_cycle = 100

# evaluate_network
EN_GAME_COUNT = 160  # 평가 1회 당 게임 수 (오리지널: 400)
EN_TEMPERATURE = 0.1  # 볼츠만 분포 온도
EN_AVERAGE_POINT = 0.5
EN_NUM_CORES = 8
