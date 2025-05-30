from itertools import chain
import os
import torch
import torch.nn as nn
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_DevTest_Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch
from evaluate import scoring
from tqdm import tqdm

from config import Config

def compute_scores(model: nn.Module,
                   mind_corpus: MIND_Corpus,
                   config: Config,
                   batch_size: int,
                   num_workers: int,
                   mode: str,
                   result_file: str,
                   dataset: str):
    assert mode in ['dev', 'test'], "mode must be chosen from 'dev' or 'test'"

    def collate_fn_eval(batch):
        samples = [b[:-4] for b in batch]
        collated = default_collate(samples)

        hist_lists, hist_masks_lists, cand_lists, cand_masks_lists = zip(*[b[-4:] for b in batch])

        flat_hists = [g for graphs in hist_lists for g in graphs]
        flat_hist_masks = [m.view(-1) if isinstance(m, torch.Tensor) else torch.as_tensor(m, dtype=torch.bool).view(-1)
                           for masks in hist_masks_lists for m in masks]

        flat_cands = [g for graphs in cand_lists for g in graphs]
        flat_cand_masks = [m.view(-1) if isinstance(m, torch.Tensor) else torch.as_tensor(m, dtype=torch.bool).view(-1)
                           for masks in cand_masks_lists for m in masks]

        hist_batch = Batch.from_data_list(flat_hists)
        cand_batch = Batch.from_data_list(flat_cands)

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

    dataloader = DataLoader(
        MIND_DevTest_Dataset(mind_corpus, config, mode),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_eval,
        num_workers=num_workers,
        pin_memory=True
    )

    # 결과를 GPU 에 올려놓고
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = mind_corpus.dev_indices if mode=='dev' else mind_corpus.test_indices
    scores  = torch.zeros(len(indices), device=device)
    index = 0

    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        for batch in (dataloader):
            # unpack
            *tensor_fields, hist_batch, hist_seed_mask, cand_batch, cand_seed_mask = batch

            # 1) 일반 텐서들만 device 로
            tensor_fields = [
                t.to(device, non_blocking=True) if hasattr(t, 'to') else t
                for t in tensor_fields
            ]
            # 2) PyG Batch 들도 device 로
            hist_batch = hist_batch.to(device)
            hist_seed_mask = hist_seed_mask.to(device)
            cand_batch    = cand_batch.to(device)
            cand_seed_mask = cand_seed_mask.to(device)

            # 3) tensor_fields 에서 다시 풀기
            (
                user_ID,
                user_category, user_subCategory,
                user_title_text, user_title_mask,
                user_history_mask,
                pretrain_hist_emb, pretrain_cand_emb,
                hist_ent_emb, hist_ent_mask,
                cand_ent_emb, cand_ent_mask,
                news_category, news_subCategory,
                news_title_text, news_title_mask,
            ) = tensor_fields

            result = model(
                user_category, user_subCategory,
                user_title_text, user_title_mask,
                user_history_mask,
                pretrain_hist_emb, pretrain_cand_emb,
                hist_ent_emb, hist_ent_mask,
                cand_ent_emb, cand_ent_mask,
                news_category, news_subCategory,
                news_title_text, news_title_mask,
                hist_batch, hist_seed_mask,
                cand_batch, cand_seed_mask
            ).squeeze(dim=1)

            batch_size = user_ID.size(0) # [batch_size, news_num]

            scores[index: index + batch_size] = result
            index += batch_size

    scores = scores.tolist()

    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, idx in enumerate(indices):
        sub_scores[idx].append([scores[i], len(sub_scores[idx])])

    with open(result_file, 'w', encoding='utf-8') as f:
        for i, ss in enumerate(sub_scores):
            ss.sort(key=lambda x: x[0], reverse=True)
            # ranks 계산
            ranks = [0] * len(ss)
            for rank, (_, pos) in enumerate(ss, 1):
                ranks[pos] = rank
            ranks_str = ','.join(str(r) for r in ranks)
            line = f"{i+1} [{ranks_str}]"
            f.write(line + ("\n" if i + 1 < len(sub_scores) else ""))

    print('result_file', result_file)
    print('save done')

    # 평가 지표 계산
    if dataset != 'large' or mode != 'test':
        with open(f"{mode}/ref/truth-{dataset}.txt", 'r', encoding='utf-8') as truth_f, \
             open(result_file, 'r', encoding='utf-8') as res_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, res_f)
        return auc, mrr, ndcg5, ndcg10
    else:
        return None, None, None, None


def get_run_index(result_dir: str):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for fname in os.listdir(result_dir):
        if fname.startswith('#') and fname.endswith('-dev'):
            idx = int(fname[1:-4])
            max_index = max(max_index, idx)
    with open(os.path.join(result_dir, f"#{max_index+1}-dev"), 'w', encoding='utf-8'):
        pass
    return max_index + 1


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, other):
        return self.avg > other.avg

    def __ge__(self, other):
        return self.avg >= other.avg

    def __lt__(self, other):
        return self.avg < other.avg

    def __le__(self, other):
        return self.avg <= other.avg

    def __str__(self):
        return (f"{self.avg:.4f}\n"
                f"AUC = {self.auc:.4f}\n"
                f"MRR = {self.mrr:.4f}\n"
                f"nDCG@5 = {self.ndcg5:.4f}\n"
                f"nDCG@10 = {self.ndcg10:.4f}")
