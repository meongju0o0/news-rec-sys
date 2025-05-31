import os
import gc
import shutil
import time
from config import Config
import torch
from MIND_corpus import MIND_Corpus
from model import Model
from trainer import Trainer
from util import compute_scores, get_run_index
import torch.multiprocessing as mp


def train(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    model.initialize()
    run_index = get_run_index(config.result_dir)
    if config.world_size == 1:
        trainer = Trainer(model, config, mind_corpus, run_index)
        trainer.train()
        trainer = None
        del trainer
    config.run_index = run_index
    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()


def dev(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, mind_corpus, config, config.batch_size, config.num_workers, 'dev', dev_res_dir + '/' + model.model_name + '.txt', config.dataset)
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10


def test(config: Config, mind_corpus: MIND_Corpus):
    start_time = time.time()
    
    model = Model(config)
    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_')).replace('.', '_')
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + os.path.join(test_res_dir, model.model_name + '.txt'))
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, mind_corpus, config, config.batch_size, config.num_workers, 'test', os.path.join(test_res_dir, model.model_name + '.txt'), config.dataset)
    if config.dataset != 'large':
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
        if config.mode == 'train':
            with open(os.path.join(config.result_dir, f'run_{config.run_index}-test'), 'w') as result_f:
                result_f.write(f'run_{config.run_index}\t{auc}\t{mrr}\t{ndcg5}\t{ndcg10}\n')
        elif config.mode == 'test' and config.test_output_file != '':
            with open(config.test_output_file, 'w', encoding='utf-8') as f:
                f.write(f'run_{config.seed + 1}\t{auc}\t{mrr}\t{ndcg5}\t{ndcg10}\n')
    else:
        if config.mode == 'train':
            shutil.copy(os.path.join(test_res_dir, model.model_name + '.txt'),
                        os.path.join('prediction/large', model.model_name, f'run_{config.run_index}', 'prediction.txt'))
            os.chdir(os.path.join('prediction/large', model.model_name, f'run_{config.run_index}'))
            os.system('zip prediction.zip prediction.txt')
            os.chdir('../../../..')
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f'time taken for test: {elapsed_time_str}')

    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print('******** start config ********')
    config = Config()

    print('******** start mind_corpus ********')
    mind_corpus = MIND_Corpus(config)

    if config.mode == 'train':
        print('******** start train ********')
        train(config, mind_corpus)
        config.test_model_path = os.path.join(config.best_model_dir, "run_" + str(config.run_index), config.news_encoder + '-' + config.user_encoder) + ".pt"
        test(config, mind_corpus)
    elif config.mode == 'dev':
        print('******** start dev ********')
        dev(config, mind_corpus)
    elif config.mode == 'test':
        print('******** start test ********')
        test(config, mind_corpus)
