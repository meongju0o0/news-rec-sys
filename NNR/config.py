import os
import argparse
import time
import torch
import random
import numpy as np
import json

class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='Neural news recommendation')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='CNE', choices=['CNE', 'CNN', 'MHSA', 'KCNN', 'HDC', 'NAML', 'PNE', 'DAE', 'Inception', 'NAML_Title', 'NAML_Content', 'CNE_Title', 'CNE_Content', 'CNE_wo_CS', 'CNE_wo_CA'], help='News encoder')
        parser.add_argument('--user_encoder', type=str, default='SUE', choices=['SUE', 'LSTUR', 'MHSA', 'ATT', 'CATT', 'FIM', 'PUE', 'GRU', 'OMAP', 'SUE_wo_GCN', 'SUE_wo_HCA'], help='User encoder')
        parser.add_argument('--dev_model_path', type=str, default='', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Specific test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--config_file', type=str, default='', help='Config file path')
        # Dataset config (지원: small, large)
        parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large', '200k'], help='Dataset type')
        parser.add_argument('--tokenizer', type=str, default='MIND', choices=['MIND', 'NLTK'], help='Sentence tokenizer')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        parser.add_argument('--max_abstract_length', type=int, default=128, help='Sentence truncate length for abstract')
        parser.add_argument('--max_linked_entity_length', type=int, default=20, help='Max linked entity length')
        parser.add_argument('--pretrain_emb_d', type=int, default=4096, help='LLM embedding dimension')
        parser.add_argument('--pretrain_rep_d', type=int, default=500, help='LLM representation dimension')
        parser.add_argument('--pretrain_hidden_dim', type=int, default=1024, help='Pretrain hidden dimension')
        parser.add_argument('--entity_hidden_dim', type=int, default=256, help='Entity hidden dimension')
        parser.add_argument('--entity_att_head_num', type=int, default=3, help='Entity attention head number')
        # Training config
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number per positive sample')
        parser.add_argument('--max_history_num', type=int, default=50, help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=32, help='Training epochs')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Optimizer weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=4, help='Gradient clip norm (non-positive for no clipping)')
        parser.add_argument('--world_size', type=int, default=1, help='World size for multi-process GPU training')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg', choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Validation criterion for model selection')
        parser.add_argument('--early_stopping_epoch', type=int, default=3, help='Epochs to stop training if dev result does not improve')
        # Model config
        parser.add_argument('--graph_entity_embedding_dim', type=int, default=100, help='Graph entity embedding dimension')
        parser.add_argument('--graph_entity_hidden_dim', type=int, default=100, help='Graph entity hidden dimension')
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--entity_embedding_dim', type=int, default=100, choices=[100], help='Entity embedding dimension')
        parser.add_argument('--context_embedding_dim', type=int, default=100, choices=[100], help='Context embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'], help='CNN group method')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernels')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size for CNN kernels')
        parser.add_argument('--attention_dim', type=int, default=200, help='Attention dimension')
        parser.add_argument('--entity_attention_dim', type=int, default=100, help='Entity attention dimension')
        parser.add_argument('--head_num', type=int, default=20, help='Number of heads in multi-head self-attention')
        parser.add_argument('--head_dim', type=int, default=20, help='Dimension of each attention head')
        parser.add_argument('--user_embedding_dim', type=int, default=50, help='User embedding dimension')
        parser.add_argument('--category_embedding_dim', type=int, default=50, help='Category embedding dimension')
        parser.add_argument('--subCategory_embedding_dim', type=int, default=50, help='SubCategory embedding dimension')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--no_self_connection', action='store_true', help='Disable self-connection in graph')
        parser.add_argument('--no_adjacent_normalization', action='store_true', help='Disable adjacent normalization in graph')
        parser.add_argument('--gcn_normalization_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'], help='GCN normalization type for adjacent matrix')
        parser.add_argument('--gcn_layer_num', type=int, default=4, help='Number of GCN layers')
        parser.add_argument('--no_gcn_residual', action='store_true', help='Disable residual connection in GCN')
        parser.add_argument('--gcn_layer_norm', action='store_true', help='Apply layer normalization in GCN')
        parser.add_argument('--hidden_dim', type=int, default=200, help='Encoder hidden dimension')
        parser.add_argument('--Alpha', type=float, default=0.1, help='Reconstruction loss weight for DAE')
        parser.add_argument('--long_term_masking_probability', type=float, default=0.1, help='Masking probability for long-term representation in LSTUR')
        parser.add_argument('--personalized_embedding_dim', type=int, default=200, help='Personalized embedding dimension for NPA')
        parser.add_argument('--HDC_window_size', type=int, default=3, help='Window size for HDC in FIM')
        parser.add_argument('--HDC_filter_num', type=int, default=150, help='Filter number for HDC in FIM')
        parser.add_argument('--conv3D_filter_num_first', type=int, default=32, help='First layer 3D convolution filter number for FIM')
        parser.add_argument('--conv3D_kernel_size_first', type=int, default=3, help='First layer 3D convolution kernel size for FIM')
        parser.add_argument('--conv3D_filter_num_second', type=int, default=16, help='Second layer 3D convolution filter number for FIM')
        parser.add_argument('--conv3D_kernel_size_second', type=int, default=3, help='Second layer 3D convolution kernel size for FIM')
        parser.add_argument('--maxpooling3D_size', type=int, default=3, help='3D pooling size for FIM')
        parser.add_argument('--maxpooling3D_stride', type=int, default=3, help='3D pooling stride for FIM')
        parser.add_argument('--OMAP_head_num', type=int, default=3, help='Head number for OMAP in Hi-Fi Ark')
        parser.add_argument('--HiFi_Ark_regularizer_coefficient', type=float, default=0.1, help='Regularizer coefficient for Hi-Fi Ark')
        parser.add_argument('--click_predictor', type=str, default='dot_product', choices=['dot_product', 'mlp', 'sigmoid', 'FIM'], help='Click predictor type')
        
        self.attribute_dict = vars(parser.parse_args())
        for key, value in self.attribute_dict.items():
            setattr(self, key, value)
        
        # 프로젝트 루트는 NNR 폴더 상위로 설정
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # 데이터셋 경로 설정 (dataset 폴더는 프로젝트 루트에 존재)
        if self.dataset == 'small':
            self.train_root = os.path.join(BASE_DIR, 'dataset', 'mind_small_preprocessed', 'train')
            self.dev_root = os.path.join(BASE_DIR, 'dataset', 'mind_small_preprocessed', 'dev')
            self.test_root = os.path.join(BASE_DIR, 'dataset', 'mind_small_preprocessed', 'test')
        elif self.dataset == 'large':
            self.train_root = os.path.join(BASE_DIR, 'dataset', 'mind_large_train')
            self.dev_root = os.path.join(BASE_DIR, 'dataset', 'mind_large_dev')
            self.test_root = os.path.join(BASE_DIR, 'dataset', 'mind_large_test')
        elif self.dataset == '200k':
            self.train_root = os.path.join(BASE_DIR, 'dataset', 'mind_200k_train')
            self.dev_root = os.path.join(BASE_DIR, 'dataset', 'mind_200k_dev')
            self.test_root = os.path.join(BASE_DIR, 'dataset', 'mind_200k_test')
        
        if self.dataset == 'small': # suggested configuration for MIND-small
            self.dropout_rate = 0.25
            self.gcn_layer_num = 3
        elif self.dataset == '200k': # suggested configuration for MIND-200k
            self.dropout_rate = 0.2
            self.gcn_layer_num = 4
            self.epoch = 8
        else: # suggested configuration for MIND-large
            self.dropout_rate = 0.1
            self.gcn_layer_num = 4
            self.epoch = 6
        
        self.seed = self.seed if self.seed >= 0 else int(time.time())
        self.attribute_dict['dropout_rate'] = self.dropout_rate
        self.attribute_dict['gcn_layer_num'] = self.gcn_layer_num
        self.attribute_dict['epoch'] = self.epoch
        self.attribute_dict['seed'] = self.seed
        
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file: ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    for key in self.attribute_dict:
                        if key in configs:
                            setattr(self, key, configs[key])
                            self.attribute_dict[key] = configs[key]
            else:
                raise Exception('Config file does not exist: ' + self.config_file)
        
        assert not (self.no_self_connection and not self.no_adjacent_normalization), \
            'Adjacent normalization can only be set when self-connection is enabled'
        for key, value in self.attribute_dict.items():
            print(key + ' : ' + str(value))
        
        assert self.batch_size % self.world_size == 0, 'For multi-GPU training, batch size must be divisible by world size'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1024'
    
    
    def set_cuda(self):
        if torch.cuda.is_available():
            print("CUDA is available, using CUDA.")
            self.device = torch.device("cuda:{}".format(self.device_id))
            torch.cuda.set_device(self.device_id)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True  # For reproducibility
        else:
            print("CUDA is not available, using CPU.")
            self.device = torch.device("cpu")
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

    
    def preliminary_setup(self):
        dataset_files = [
            os.path.join(self.train_root, 'news.tsv'),
            os.path.join(self.train_root, 'behaviors.tsv'),
            os.path.join(self.train_root, 'entity_embedding.vec'),
            os.path.join(self.dev_root, 'news.tsv'),
            os.path.join(self.dev_root, 'behaviors.tsv'),
            os.path.join(self.dev_root, 'entity_embedding.vec'),
            os.path.join(self.test_root, 'news.tsv'),
            os.path.join(self.test_root, 'behaviors.tsv'),
            os.path.join(self.test_root, 'entity_embedding.vec')
        ]
        
        # 데이터셋 파일 중 하나라도 없으면, 해당 prepare 함수를 호출하여 전처리 진행
        if not all(os.path.exists(f) for f in dataset_files):
            if self.dataset == 'small':
                from prepare_MIND_dataset import prepare_MIND_small
                prepare_MIND_small()
            elif self.dataset == 'large':
                from prepare_MIND_dataset import prepare_MIND_large
                prepare_MIND_large()
        
        model_name = self.news_encoder + '-' + self.user_encoder
        mkdirs = lambda x: os.makedirs(x) if not os.path.exists(x) else None
        self.config_dir = os.path.join('configs', self.dataset, model_name)
        self.model_dir = os.path.join('models', self.dataset, model_name)
        self.best_model_dir = os.path.join('best_model', self.dataset, model_name)
        self.dev_res_dir = os.path.join('dev', 'res', self.dataset, model_name)
        self.test_res_dir = os.path.join('test', 'res', self.dataset, model_name)
        self.result_dir = os.path.join('results', self.dataset, model_name)
        
        mkdirs(self.config_dir)
        mkdirs(self.model_dir)
        mkdirs(self.best_model_dir)
        mkdirs(os.path.join('dev', 'ref'))
        mkdirs(self.dev_res_dir)
        mkdirs(os.path.join('test', 'ref'))
        mkdirs(self.test_res_dir)
        mkdirs(self.result_dir)
        
        if not os.path.exists(os.path.join('dev', 'ref', f'truth-{self.dataset}.txt')):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open(os.path.join('dev', 'ref', f'truth-{self.dataset}.txt'), 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if self.dataset != 'large':
            if not os.path.exists(os.path.join('test', 'ref', f'truth-{self.dataset}.txt')):
                with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                    with open(os.path.join('test', 'ref', f'truth-{self.dataset}.txt'), 'w', encoding='utf-8') as truth_f:
                        for test_ID, line in enumerate(test_f):
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        else:
            self.prediction_dir = os.path.join('prediction', 'large', model_name)
            mkdirs(self.prediction_dir)
    
    
    def __init__(self):
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        self.parse_argument()
        self.preliminary_setup()
        self.set_cuda()
        print('*' * 32 + ' Experiment setting ' + '*' * 32)


if __name__ == '__main__':
    config = Config()
