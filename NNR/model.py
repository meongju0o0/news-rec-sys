from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import newsEncoders
import userEncoders
import variantEncoders


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        # For main experiments of news encoding
        if config.news_encoder == 'MHSA':
            self.news_encoder = newsEncoders.MHSA(config)
        else:
            raise Exception(config.news_encoder + 'is not implemented')

        # For main experiments of user encoding
        if config.user_encoder == 'MHSA':
            self.user_encoder = userEncoders.MHSA(self.news_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')

        self.model_name = config.news_encoder + '-' + config.user_encoder
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.click_predictor = config.click_predictor

        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.news_embedding_dim * 2, out_features=self.news_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.news_embedding_dim // 2, out_features=1, bias=True)
        elif self.click_predictor == 'FIM':
            # compute the output size of 3D convolution and pooling
            def compute_convolution_pooling_output_size(input_size):
                conv1_size = input_size - config.conv3D_kernel_size_first + 1
                pool1_size = (conv1_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
                conv2_size = pool1_size - config.conv3D_kernel_size_second + 1
                pool2_size = (conv2_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
                return pool2_size
            feature_size = compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                           compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                           compute_convolution_pooling_output_size(config.max_history_num) * \
                           config.conv3D_filter_num_second
            self.fc = nn.Linear(in_features=feature_size, out_features=1, bias=True)

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()

        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)
        elif self.click_predictor == 'FIM':
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self,
                user_category, user_subCategory,
                user_title_text, user_title_mask,
                user_history_mask,
                pretrain_history_emb, pretrain_candidate_emb,
                hist_ent_emb, hist_ent_mask, cand_ent_emb, cand_ent_mask,
                news_category, news_subCategory,
                news_title_text, news_title_mask,
                hist_batch, hist_seed_mask,
                cand_batch, cand_seed_mask
    ):
        news_representation = self.news_encoder(
            news_title_text, news_title_mask,
            news_category, news_subCategory,
            pretrain_candidate_emb,
            cand_ent_emb, cand_ent_mask,
            cand_batch, cand_seed_mask
        ) # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        
        user_representation = self.user_encoder(
            user_title_text, user_title_mask,
            user_category, user_subCategory,
            user_history_mask,
            news_representation,
            pretrain_history_emb,
            hist_ent_emb, hist_ent_mask,
            hist_batch, hist_seed_mask
        ) # [batch_size, 1 + negative_sample_num, news_embedding_dim]

        if self.click_predictor == 'dot_product':
            logits = (user_representation * news_representation).sum(dim=2) # dot-product  torch.Size([64, 5])
        elif self.click_predictor == 'mlp':
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, news_representation], dim=2)), inplace=True))
            logits = self.out(context).squeeze(dim=2)
        elif self.click_predictor == 'FIM':
            logits = self.fc(user_representation).squeeze(dim=2)
        return logits
