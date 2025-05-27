import os
import warnings
import argparse
import pickle
import torch
from transformers import logging
from transformers import AutoTokenizer, AutoModel

from preprocess_news import load_news_data, construct_news_text

warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable_progress_bar()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("ERROR")


def load_model_and_tokenizer(model_name="THUDM/chatglm2-6b"):
    """
    주어진 모델명을 기반으로 토크나이저와 모델을 로드
    """
    print(">> 모델과 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./chat_glm_model', trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, cache_dir='./chat_glm_model', trust_remote_code=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">> 사용 디바이스:", device)
    model = model.to(device)
    
    # 파인튜닝 없이 임베딩 추출을 위해 파라미터 고정
    for param in model.parameters():
        param.requires_grad = False

    model.config.output_hidden_states = True
    print(">> 모델과 토크나이저 로드 완료.")
    return tokenizer, model, device


def compute_news_embeddings(itemid_list, itemtext_list, tokenizer, model, device, 
                            print_every=100, output_file="item_emb.pkl"):
    """
    - 각 뉴스 텍스트에 대해 LLM 임베딩을 계산하여 단일 임베딩 벡터를 생성
    - 최종적으로 전체 임베딩 딕셔너리를 output_file에 저장
    
    단계:
      1. LLM의 마지막 4개 레이어의 hidden states 추출
      2. 각 레이어에 대해 토큰 차원(mean pooling)을 적용하여 (hidden_dim,) 벡터 도출  
         - 입력이 여러 청크(B > 1)라면, 청크 차원에 대해서도 평균을 수행하여 단일 벡터 생성
      3. 네 레이어의 평균 벡터들을 단순 평균하여 최종 임베딩 산출

    최종 임베딩은 numpy 배열로 변환되어 하나의 딕셔너리로 저장
    """
    item_count = len(itemtext_list)
    embeddings = {}
    print(">> 뉴스 텍스트 임베딩 계산 시작...")
    
    model.eval()
    with torch.no_grad():
        for idx, item_text in enumerate(itemtext_list):
            itemid = itemid_list[idx]
            # tokenizer의 결과 딕셔너리의 모든 텐서를 device로 이동
            inputs = tokenizer.encode_plus(item_text, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            # 마지막 4개 레이어의 hidden states 추출 (각 텐서: [B, seq_length, hidden_dim])  
            last_four = outputs.hidden_states[-4:]
            layer_means = []
            for layer in last_four:
                # 토큰 차원에 대해 평균 풀링 => (B, hidden_dim)
                token_mean = layer.mean(dim=1)
                # 만약 배치(B)크기가 1보다 크다면, 배치 차원에 대해서도 평균내어 (1, hidden_dim) 생성
                overall_mean = token_mean.mean(dim=0, keepdim=True)
                layer_means.append(overall_mean)
            # (4, hidden_dim) 텐서를 단순 평균하여 최종 임베딩 (hidden_dim,) 산출
            stacked = torch.cat(layer_means, dim=0)    # shape: (4, hidden_dim)
            final_embedding = stacked.mean(dim=0)      # shape: (hidden_dim,)
            
            embedding_array = final_embedding.detach().cpu().numpy()
            embeddings[itemid] = embedding_array

            if (idx + 1) % print_every == 0:
                print(f"   처리 완료: {idx+1}/{item_count}, 임베딩 차원: {embedding_array.shape}")

    # 최종 딕셔너리를 pickle 파일로 저장
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(">> 뉴스 텍스트 임베딩 계산 완료.")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="News Embedding Computation")
    parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large', '200k'],
                            help='Dataset type: small or large')
    args = parser.parse_args()

    news_file = f"../dataset/mind_{args.dataset}_merged/news.tsv"

    news = load_news_data(news_file)
    news = construct_news_text(news)
    itemid_list = news["itemID"].tolist()
    itemtext_list = news["news_text"].tolist()

    tokenizer, model, device = load_model_and_tokenizer("THUDM/chatglm2-6b")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    output_file = f"{args.dataset}/item_emb.pkl"
    if os.path.exists(output_file):
        os.remove(output_file)

    compute_news_embeddings(itemid_list, itemtext_list, tokenizer, model, device, print_every=100, output_file=output_file)
    print(">> 모든 임베딩 저장 완료.")