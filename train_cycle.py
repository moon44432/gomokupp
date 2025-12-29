import torch
from network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate import evaluate_network
from record_play import generate_record_list
from record_play import record_play
from hparams import record_batch_size, train_cycle


if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    dual_network()

    CALL_COUNT = 20
    record_list = generate_record_list()

    for i in range(train_cycle):
        print('Training {:04d}'.format(i + 1))

        # 기보로 학습 데이터 생성 파트

        print('Record count: ', len(record_list))
        if CALL_COUNT * record_batch_size >= len(record_list):
            CALL_COUNT = 0

        print('Record {}~{}'.format(CALL_COUNT * record_batch_size + 1, (CALL_COUNT + 1) * record_batch_size))
        # 기보 데이터
        record = record_list[CALL_COUNT * record_batch_size:(CALL_COUNT + 1) * record_batch_size]

        print('Generating data from records...')
        record_play(record)
        CALL_COUNT += 1


        # 셀프 플레이로 데이터 생성 파트
        #print('Generating data from self playing...')
        #self_play()

        # 파라미터 갱신 파트
        print('Training network...')
        train_network()

        # 신규 파라미터 평가 파트
        print('Evaluating network...')
        evaluate_network()
