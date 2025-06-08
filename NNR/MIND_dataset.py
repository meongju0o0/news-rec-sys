import os
import pickle
import random
import numpy as np
import torch
import torch.utils.data as data
from MIND_corpus import MIND_Corpus

from config import Config


class BaseMINDDataset(data.Dataset):
    """
    Base loader for MIND datasets. Loads news meta, user history graphs,
    pre-trained embeddings and linked entity embeddings.
    """
    _global_pretrain_emb = None
    _global_linked_entity_emb = None
    _global_linked_entity_mask = None
    _global_news_subgraphs = None
    _global_news_subgraph_masks = None

    def __init__(self, corpus: MIND_Corpus, config: Config, mode: str):
        assert mode in ['train', 'dev', 'test'], "mode must be 'train', 'dev', or 'test'"
        self.mode = mode

        # share news features
        self.news_category      = torch.from_numpy(corpus.news_category)
        self.news_subCategory   = torch.from_numpy(corpus.news_subCategory)
        self.news_title_text    = torch.from_numpy(corpus.news_title_text)
        self.news_title_mask    = torch.from_numpy(corpus.news_title_mask)

        self.pretrain_emb_d = corpus.pretrain_emb_d
        self.max_linked_entity_length = config.max_linked_entity_length
        self.entity_embedding_dim = config.entity_embedding_dim
        self.graph_entity_dim = config.gat_in_dim

        # share history graphs and behaviors
        if mode == 'train':
            self.behaviors                     = corpus.train_behaviors
            self.negative_sample_num           = corpus.negative_sample_num
            self.samples                       = [[0]*(1+self.negative_sample_num) for _ in self.behaviors]
        elif mode == 'dev':
            self.behaviors                     = corpus.dev_behaviors
        else:
            self.behaviors                     = corpus.test_behaviors

        self.num = len(self.behaviors)

        # build ID <-> news mapping for linked entities
        self.news_id_mapping = corpus.news_ID_dict
        self.id_news_mapping = {v:k for k,v in self.news_id_mapping.items()}

        # load pretrain embeddings and entity dictionary once
        self._load_pretrain_emb()
        self._load_linked_entity_emb()

        # load news subgraph once
        self._load_news_subgraph(config)

    def _load_pretrain_emb(self):
        if BaseMINDDataset._global_pretrain_emb is None:
            emb_dim = self.pretrain_emb_d
            tmp = torch.zeros((len(self.news_id_mapping), emb_dim), dtype=torch.float32)
            item_emb = np.load('./item_emb.pkl', allow_pickle=True)
            assert isinstance(item_emb, dict)
            for news_id, idx in self.news_id_mapping.items():
                if news_id=='<PAD>':
                    continue
                vec = item_emb.get(news_id)
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                assert arr.shape[0] == emb_dim
                tmp[idx] = torch.from_numpy(arr)
            BaseMINDDataset._global_pretrain_emb = tmp
        
        self.pretrain_emb = BaseMINDDataset._global_pretrain_emb

    def _load_linked_entity_emb(self):
        if BaseMINDDataset._global_linked_entity_emb is None or BaseMINDDataset._global_linked_entity_mask is None:
            N = len(self.news_id_mapping)
            L = self.max_linked_entity_length
            D = self.entity_embedding_dim
            tmp_emb = np.zeros((N, L, D), dtype=np.float32)
            tmp_mask = np.zeros((N, L), dtype=bool)

            with open('../graph/link_entity_dic.pkl', 'rb') as f:
                link_entity_dic = pickle.load(f)
            with open('../graph/all_entity_emb_dic.pkl', 'rb') as f:
                ent_emb_dic = pickle.load(f)

            for idx in range(N):
                news_id = self.id_news_mapping[idx]
                if news_id == '<PAD>': 
                    continue
                linked_entity = link_entity_dic.get(news_id, [])
                mask_len = min(len(linked_entity), L)
                tmp_mask[idx, :mask_len] = True
                if len(linked_entity) > L:
                    linked_entity = random.sample(linked_entity, L)
                for j, ent in enumerate(linked_entity[:L]):
                    tmp_emb[idx, j] = ent_emb_dic[ent]
            
            BaseMINDDataset._global_linked_entity_emb = torch.from_numpy(tmp_emb)
            BaseMINDDataset._global_linked_entity_mask = torch.from_numpy(tmp_mask)

        self.linked_entity_emb = BaseMINDDataset._global_linked_entity_emb
        self.linked_entity_mask = BaseMINDDataset._global_linked_entity_mask
    
    def _load_news_subgraph(self, config: Config):
        if BaseMINDDataset._global_news_subgraphs is None:
            with open(f'news_subgraph-{config.dataset}.pkl', 'rb') as f:
                graphs, masks = zip(*pickle.load(f))

            BaseMINDDataset._global_news_subgraphs = graphs
            BaseMINDDataset._global_news_subgraph_masks = masks
        
        self.news_subgraphs = BaseMINDDataset._global_news_subgraphs
        self.news_subgraph_masks = BaseMINDDataset._global_news_subgraph_masks

    def __len__(self):
        return self.num


class MIND_Train_Dataset(BaseMINDDataset):
    def __init__(self, corpus: MIND_Corpus, config: Config):
        super().__init__(corpus, config, mode='train')
        self.negative_sample_num = corpus.negative_sample_num
    
    def negative_sampling(self, rank=None):
        if self.mode!='train': return
        print(f"Begin negative sampling, num={self.num}")
        for i, beh in enumerate(self.behaviors):
            self.samples[i][0] = beh[3]
            negs = beh[4]
            if len(negs)==0: raise ValueError("no negatives")
            if len(negs)<=self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.samples[i][j+1] = negs[j%len(negs)]
            else:
                self.samples[i][1:] = random.sample(negs, self.negative_sample_num)
        print("Negative sampling done")

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_idx = torch.tensor(behavior[1])
        sample_idx = torch.tensor(self.samples[index])

        # pretrain embeddings split
        pretrain_hist = self.pretrain_emb[history_idx]    # [max_history, emb_d]
        pretrain_cand = self.pretrain_emb[sample_idx]     # [1+neg, emb_d]
        # entity embeddings split
        hist_ent_emb = self.linked_entity_emb[history_idx]
        hist_ent_mask = self.linked_entity_mask[history_idx]
        cand_ent_emb = self.linked_entity_emb[sample_idx]
        cand_ent_mask = self.linked_entity_mask[sample_idx]

        history_graphs = [self.news_subgraphs[i] for i in history_idx.tolist()]
        history_seed_masks = [self.news_subgraph_masks[i] for i in history_idx.tolist()]
        cand_graphs = [self.news_subgraphs[i] for i in sample_idx.tolist()]
        cand_seed_masks = [self.news_subgraph_masks[i] for i in sample_idx.tolist()]

        return (
            behavior[0],  # user ID
            # history news meta
            self.news_category[history_idx],
            self.news_subCategory[history_idx],
            self.news_title_text[history_idx],
            self.news_title_mask[history_idx],
            behavior[2],
            # pretrain embeddings
            pretrain_hist,
            pretrain_cand,
            # entity embeddings
            hist_ent_emb,
            hist_ent_mask,
            cand_ent_emb,
            cand_ent_mask,
            # candidate news meta
            self.news_category[sample_idx],
            self.news_subCategory[sample_idx],
            self.news_title_text[sample_idx],
            self.news_title_mask[sample_idx],
            # news subgraph
            history_graphs, history_seed_masks,
            cand_graphs, cand_seed_masks
        )


class MIND_DevTest_Dataset(BaseMINDDataset):
    """
    Dev/Test dataset: each behavior has exactly one candidate.
    """
    def __init__(self, corpus: MIND_Corpus, config: Config, mode: str):
        super().__init__(corpus, config, mode=mode)

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_idx = torch.tensor(behavior[1])
        cand_news_idx = torch.tensor(behavior[3])

        pretrain_hist = self.pretrain_emb[history_idx]
        pretrain_cand = self.pretrain_emb[cand_news_idx]
        hist_ent_emb = self.linked_entity_emb[history_idx]
        hist_ent_mask = self.linked_entity_mask[history_idx]
        cand_ent_emb = self.linked_entity_emb[cand_news_idx]
        cand_ent_mask = self.linked_entity_mask[cand_news_idx]

        history_graphs = [self.news_subgraphs[i] for i in history_idx.tolist()]
        history_seed_masks = [self.news_subgraph_masks[i] for i in history_idx.tolist()]
        cand_graphs = [self.news_subgraphs[cand_news_idx.item()]]
        cand_seed_masks = [self.news_subgraph_masks[cand_news_idx.item()]]

        return (
            behavior[0],  # user ID
            # history news meta
            self.news_category[history_idx],
            self.news_subCategory[history_idx],
            self.news_title_text[history_idx],
            self.news_title_mask[history_idx],
            behavior[2],
            # pretrain embeddings
            pretrain_hist,
            pretrain_cand.unsqueeze(dim=0),
            # entity embeddings
            hist_ent_emb,
            hist_ent_mask,
            cand_ent_emb,
            cand_ent_mask,
            # candidate news meta
            self.news_category[cand_news_idx].unsqueeze(dim=0),
            self.news_subCategory[cand_news_idx].unsqueeze(dim=0),
            self.news_title_text[cand_news_idx].unsqueeze(dim=0),
            self.news_title_mask[cand_news_idx].unsqueeze(dim=0),
            # news subgraph
            history_graphs, history_seed_masks,
            cand_graphs, cand_seed_masks
        )
