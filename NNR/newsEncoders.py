import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import json
from layers import GAT, MultiHeadAttention, Attention, QAttention


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None
        

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, pretrain_emb_batch, sample_index):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory, pretrain_emb, batch_linked_entity_emb_sum, gat_rep):
        # if pretrain_emb:
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                     # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation), pretrain_emb, batch_linked_entity_emb_sum, gat_rep], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation
        # else:
        #     category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        #     subCategory_representation = self.subCategory_embedding(subCategory)                                                                     # [batch_size, news_num, subCategory_embedding_dim]
        #     news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation), batch_linked_entity_emb_sum], dim=2) # [batch_size, news_num, news_embedding_dim]
        #     return news_representation


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.entity_attention = QAttention(config.entity_embedding_dim*3, config.entity_attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim + config.pretrain_rep_d + config.entity_hidden_dim
        # self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim  + config.entity_hidden_dim
        # 得到大模型emb后进行操作
        self.dense1 = nn.Linear(config.pretrain_emb_d, config.pretrain_hidden_dim, bias=True)
        self.dense2 = nn.Linear(config.pretrain_hidden_dim, config.pretrain_rep_d, bias=True)
        # 查询到entity向量后进行操作
        self.dense3 = nn.Linear(config.entity_embedding_dim*config.entity_att_head_num, config.entity_hidden_dim*2, bias=True)
        self.dense4 = nn.Linear(config.entity_hidden_dim*2,  config.entity_hidden_dim, bias=True)

        # 得到大模型emb后进行Q pre操作
        self.denseQ_pre1 = nn.Linear(config.pretrain_emb_d, config.pretrain_hidden_dim, bias=True)
        self.denseQ_pre2 = nn.Linear(config.pretrain_hidden_dim, config.pretrain_rep_d, bias=True)

        # KG多头注意力
        self.denseHead1 = nn.Linear(config.pretrain_rep_d,  config.entity_embedding_dim, bias=True)
        self.denseHead2 = nn.Linear(config.pretrain_rep_d,  config.entity_embedding_dim, bias=True)
        self.denseHead3 = nn.Linear(config.pretrain_rep_d,  config.entity_embedding_dim, bias=True)

        self.gat_num_heads = 1
        self.gat_in_channels = config.graph_entity_embedding_dim  # 768
        self.gat_hidden_channels = config.graph_entity_hidden_dim  # 384
        self.gat_out_channels = config.graph_entity_hidden_dim  # 384
        self.mlp_out_dim = 384  # GAT 후에 하나 더 거칠 MLP: graph_entity_hidden_dim → 384
        # KG embedding via GAT
        self.gat = GAT(
            in_channels=self.gat_in_channels,
            hidden_channels=self.gat_hidden_channels,
            out_channels=self.gat_out_channels,
            num_layers=2,
            dropout=0.5,
            num_heads=self.gat_num_heads
        )
        # GAT 후에 하나 더 거칠 MLP: graph_entity_hidden_dim → 384
        self.gat_mlp = nn.Sequential(
            nn.Linear(self.gat_out_channels * self.gat_num_heads, self.mlp_out_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.news_embedding_dim += self.mlp_out_dim
    
    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()
        self.entity_attention.initialize()
        self.gat.reset_parameters()
        nn.init.xavier_uniform_(self.dense1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_uniform_(self.dense2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense2.bias)
        nn.init.xavier_uniform_(self.dense3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense3.bias)
        nn.init.xavier_uniform_(self.dense4.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense4.bias)
        nn.init.xavier_uniform_(self.denseQ_pre1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseQ_pre1.bias)
        nn.init.xavier_uniform_(self.denseQ_pre2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseQ_pre2.bias)
        nn.init.xavier_uniform_(self.denseHead1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseHead1.bias)
        nn.init.xavier_uniform_(self.denseHead2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseHead2.bias)
        nn.init.xavier_uniform_(self.denseHead3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseHead3.bias)
        nn.init.xavier_uniform_(self.gat_mlp[0].weight)
        nn.init.zeros_(self.gat_mlp[0].bias)

    def forward(self,
            title_text, title_mask,
            category, subCategory,
            pretrain_emb,
            batch_linked_entity_emb, batch_linked_entity_mask,
            news_subgraph_batch, news_subgraph_entity_mask
    ):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          #[batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) #[batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                    #[batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])                           #[batch_size, news_num, news_embedding_dim]
        # 4. pretrain
        llm_original            = F.normalize(pretrain_emb, dim=-1)
        pretrain_hidden         = F.relu(self.dropout(self.dense1(llm_original)), inplace=True)
        pretrain_representation = F.relu(self.dropout(self.dense2(pretrain_hidden)), inplace=True)
        # 5. linked node emb
        # entity Query
        llm_query = F.relu(self.dropout(self.denseQ_pre1(llm_original)), inplace=True)
        llm_query = F.relu(self.dropout(self.denseQ_pre2(llm_query)), inplace=True)
        entityQuery_head1 = F.relu(self.dropout(self.denseHead1(llm_query)), inplace=True).view([batch_news_num, self.config.entity_embedding_dim])  
        entityQuery_head1 = entityQuery_head1.unsqueeze(1).expand(-1, self.config.max_linked_entity_length, -1)
        entityQuery_head2 = F.relu(self.dropout(self.denseHead2(llm_query)), inplace=True).view([batch_news_num, self.config.entity_embedding_dim])  
        entityQuery_head2 = entityQuery_head2.unsqueeze(1).expand(-1, self.config.max_linked_entity_length, -1)
        entityQuery_head3 = F.relu(self.dropout(self.denseHead3(llm_query)), inplace=True).view([batch_news_num, self.config.entity_embedding_dim])  
        entityQuery_head3 = entityQuery_head3.unsqueeze(1).expand(-1, self.config.max_linked_entity_length, -1)
        entityQuery = [entityQuery_head1, entityQuery_head2, entityQuery_head3]
        # get entity
        batch_linked_entity_mask = batch_linked_entity_mask.view([batch_news_num, self.config.max_linked_entity_length])
        batch_linked_entity_emb = batch_linked_entity_emb.view([batch_news_num, self.config.max_linked_entity_length, self.config.entity_embedding_dim])
        batch_linked_entity_emb_sum = self.entity_attention(batch_linked_entity_emb, entityQuery, mask=batch_linked_entity_mask).view([batch_size, news_num, self.config.entity_embedding_dim*self.config.entity_att_head_num]) #[batch_size, news_num, news_embedding_dim]
        batch_linked_entity_emb_sum = F.normalize(batch_linked_entity_emb_sum, dim=2)
        batch_linked_entity_emb_sum = F.relu(self.dropout(self.dense3(batch_linked_entity_emb_sum)), inplace=True)
        batch_linked_entity_emb_sum = F.relu(self.dropout(self.dense4(batch_linked_entity_emb_sum)), inplace=True)
        
        # 6. Graph embedding via GAT → Pool → MLP
        #    news_subgraph_batch.x is zero for dummy graphs,
        #    so node_emb will also be zero (up to biases), and mean pooling yields zero.
        x          = news_subgraph_batch.x
        edge_index = news_subgraph_batch.edge_index
        edge_attr  = getattr(news_subgraph_batch, 'edge_attr', None)

        node_emb, _ = self.gat(x, edge_index, edge_attr)
        batch_idx = news_subgraph_batch.batch

        masked_node_emb = node_emb[news_subgraph_entity_mask]
        masked_batch_idx = batch_idx[news_subgraph_entity_mask]

        # seed node 기준으로 mean pooling
        if masked_node_emb.size(0) > 0:
            pooled = global_mean_pool(masked_node_emb, masked_batch_idx)  # [B*N, hidden_dim]
        else:
            # 모든 마스크가 False일 경우 fallback (dummy graph)
            pooled = torch.zeros((batch_news_num, node_emb.size(1)), dtype=node_emb.dtype, device=node_emb.device)
        
        gat_rep = self.gat_mlp(pooled)  # [B*N, 384]
        gat_rep = gat_rep.view(batch_size, news_num, -1) # [B, N, 384]

        # 7. feature fusion (gat_rep 포함)
        news_representation = self.feature_fusion(
            news_representation,
            category,
            subCategory,
            pretrain_representation,
            batch_linked_entity_emb_sum,
            gat_rep
        )
        
        return news_representation
