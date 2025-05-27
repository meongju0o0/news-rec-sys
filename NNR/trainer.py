from itertools import chain
import os
import json
import time
from config import Config
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_Train_Dataset
from util import AvgMetric
from util import compute_scores
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data

class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
        self.model = model
        self.config = config
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.num_workers = config.num_workers
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=config.lr, weight_decay=config.weight_decay)
        self._dataset = config.dataset
        self.mind_corpus = mind_corpus
        self.train_dataset = MIND_Train_Dataset(mind_corpus, config)
        self.run_index = run_index

        self.model_dir = os.path.join(config.model_dir, "run_" + str(self.run_index))
        self.best_model_dir = os.path.join(config.best_model_dir, "run_" + str(self.run_index))
        self.dev_res_dir = os.path.join(config.dev_res_dir, "run_" + str(self.run_index))
        self.result_dir = config.result_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
        with open(os.path.join(config.config_dir, "run_" + str(self.run_index) + '.json'), 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if self._dataset == 'large':
            self.prediction_dir = os.path.join(config.prediction_dir, "run_" + str(self.run_index))
            os.mkdir(self.prediction_dir)

        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc_results = []
        self.mrr_results = []
        self.ndcg5_results = []
        self.ndcg10_results = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg5 = 0
        self.best_dev_ndcg10 = 0
        self.best_dev_avg = AvgMetric(0, 0, 0, 0)
        self.epoch_not_increase = 0
        self.gradient_clip_norm = config.gradient_clip_norm
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        print('Running : ' + self.model.model_name + '\trun_' + str(self.run_index))

    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss
    
    def collate_fn(self, batch):
        # 1) plain tensor 부분 모으기
        samples = [b[:-4] for b in batch]
        collated = default_collate(samples)

        # 2) 그래프와 마스크 부분 따로 모으기
        hist_lists       = [b[-4] for b in batch]   # list of list of Data
        hist_masks_lists = [b[-3] for b in batch]   # list of list of Tensor
        cand_lists       = [b[-2] for b in batch]
        cand_masks_lists = [b[-1] for b in batch]

        # 3) 한 번 더 flatten
        flat_hists      = list(chain.from_iterable(hist_lists))
        flat_hist_masks = list(chain.from_iterable(hist_masks_lists))
        flat_cands      = list(chain.from_iterable(cand_lists))
        flat_cand_masks = list(chain.from_iterable(cand_masks_lists))

        # 4) PyG Batch 생성
        hist_batch = Batch.from_data_list(flat_hists)
        cand_batch = Batch.from_data_list(flat_cands)

        # 5) mask를 1D bool Tensor로 강제 변환
        hist_mask_tensors = [
            m.view(-1) if isinstance(m, torch.Tensor)
            else torch.as_tensor(m, dtype=torch.bool).view(-1)
            for m in flat_hist_masks
        ]
        cand_mask_tensors = [
            m.view(-1) if isinstance(m, torch.Tensor)
            else torch.as_tensor(m, dtype=torch.bool).view(-1)
            for m in flat_cand_masks
        ]

        # 6) concat
        hist_seed_mask = torch.cat(hist_mask_tensors, dim=0)
        cand_seed_mask = torch.cat(cand_mask_tensors, dim=0)

        return (*collated, hist_batch, hist_seed_mask, cand_batch, cand_seed_mask)

    def train(self):
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Move model to device
        model = self.model.to(device)

        # 초기 best 성능 및 early stopping 변수 초기화 (여기서는 AUC 기준으로 설정)
        best_dev_metric = 0.0
        best_dev_epoch = 0
        epoch_not_increase = 0
        training_logs = []

        for epoch in range(1, self.epoch + 1):
            print(f'---- Starting epoch {epoch} ----')

            start_time = time.time()
            self.train_dataset.negative_sampling()

            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=True if device.type == "cuda" else False
            )
            model.train()
            epoch_loss = 0

            for batch in (train_dataloader):
                # 1) unpack collate_fn’s return:
                #    (*all_the_normal_tensors,
                #     history_batch, history_sizes)
                *tensor_fields, hist_batch, hist_seed_mask, cand_batch, cand_seed_mask = batch

                # 2) move your plain tensors to device
                tensor_fields = [
                    t.to(device, non_blocking=True) if hasattr(t, 'to') else t
                    for t in tensor_fields
                ]

                # 3) move the PyG Batch objects
                hist_batch = hist_batch.to(device)
                hist_seed_mask    = hist_seed_mask.to(device)
                cand_batch = cand_batch.to(device)
                cand_seed_mask = cand_seed_mask.to(device)

                # 4) Now unpack the moved tensor_fields back into named variables
                (
                    user_ID,
                    user_category, user_subCategory,
                    user_title_text, user_title_mask,
                    user_history_mask,
                    pretrain_history_emb, pretrain_candidate_emb,
                    hist_ent_emb, hist_ent_mask,
                    cand_ent_emb, cand_ent_mask,
                    news_category, news_subCategory,
                    news_title_text, news_title_mask
                ) = tensor_fields

                # — now you can call your model safely —
                logits = model(
                    user_category, user_subCategory,
                    user_title_text, user_title_mask,
                    user_history_mask,
                    pretrain_history_emb, pretrain_candidate_emb,
                    hist_ent_emb, hist_ent_mask,
                    cand_ent_emb, cand_ent_mask,
                    news_category, news_subCategory,
                    news_title_text, news_title_mask,
                    hist_batch, hist_seed_mask,
                    cand_batch, cand_seed_mask
                )

                # Compute loss
                loss = self.loss(logits)
                if model.news_encoder.auxiliary_loss is not None:
                    loss += model.news_encoder.auxiliary_loss.mean()
                if model.user_encoder.auxiliary_loss is not None:
                    loss += model.user_encoder.auxiliary_loss.mean()
                epoch_loss += loss.item() * user_ID.size(0)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            # Logging train loss for this epoch
            avg_loss = epoch_loss / len(self.train_dataset)
            print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')

            # Validation step (dev dataset)
            auc, mrr, ndcg5, ndcg10 = compute_scores(
                model, self.mind_corpus, self.config, 2 * self.batch_size, self.num_workers, 'dev',
                os.path.join(self.dev_res_dir, f"{model.model_name}-{epoch}.txt"), self._dataset
            )
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            end_time = time.time()

            print(f'Validation results - Epoch {epoch}: AUC={auc:.4f}, MRR={mrr:.4f}, nDCG@5={ndcg5:.4f}, nDCG@10={ndcg10:.4f}')

            # 총 소요 시간 시간, 분, 초 형식으로 출력
            elapsed_time = end_time - start_time
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f'Time taken for epoch {epoch}: {elapsed_time_str}')
            
            # (예시: AUC 기준으로 best model 저장, 다른 기준을 원하면 아래 current_metric 계산을 수정)
            current_metric = auc
            if current_metric > best_dev_metric:
                best_dev_metric = current_metric
                best_dev_epoch = epoch
                epoch_not_increase = 0
                # Best model 저장 (state_dict를 저장)
                best_model_path = os.path.join(self.best_model_dir, f"{model.model_name}.pt")
                torch.save({model.model_name: model.state_dict()}, best_model_path)
                print(f"Best model updated at epoch {epoch} with AUC {auc:.4f}. Saved to {best_model_path}")
            else:
                epoch_not_increase += 1
            # 기록용 로그 저장 (에폭별 metric 기록)
            training_logs.append(f"Epoch {epoch}:\tAUC: {auc:.4f}\tMRR: {mrr:.4f}\tnDCG@5: {ndcg5:.4f}\tnDCG@10: {ndcg10:.4f}")

            # Early stopping 조건 (예시: 5 에폭 연속 개선 없으면 중단)
            if epoch_not_increase > self.early_stopping_epoch:
                print(f"Early stopping triggered: no improvement for {self.early_stopping_epoch} consecutive epochs.")
                break

            torch.cuda.empty_cache()

        # Training 종료 후 로그 파일에 기록
        log_file_path = os.path.join(self.result_dir, f"run_{self.run_index}_train.log")
        with open(log_file_path, 'w', encoding='utf-8') as log_f:
            log_f.write("Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n")
            for log_line in training_logs:
                log_f.write(log_line + "\n")
            log_f.write(f"Best epoch: {best_dev_epoch}\tBest AUC: {best_dev_metric:.4f}\n")
        print(f"Training logs saved to {log_file_path}")
        print(f"Training completed for model {model.model_name} run_{self.run_index}")

def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss
