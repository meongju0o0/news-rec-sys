import csv
import pandas as pd


def load_news_data(news_file):
    """
    주어진 뉴스 파일(news.tsv)을 읽어 DataFrame으로 반환
    """
    print(">> 뉴스 데이터 로드 중:", news_file)
    columns = ["itemID", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    news = pd.read_csv(
        news_file, 
        sep="\t", 
        names=columns, 
        header=None, 
        engine="python", 
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip"
    )
    print(">> 뉴스 데이터 로드 완료. 총 뉴스 개수:", len(news))
    print(">> 뉴스 데이터 컬럼:")
    print(news.columns)
    print(">> 뉴스 데이터 샘플:")
    print(news.head())
    return news


def construct_news_text(news_df):
    """
    각 뉴스 항목의 title, category, abstract를 이용해 'news_text' 컬럼을 생성
    """
    def _make_text(x):
        return (
            "news title: " + str(x["title"]) + "\n" +
            "news category: " + str(x["category"]) + "\n" +
            "news abstract: " + str(x["abstract"])
        )
    
    news_df["news_text"] = news_df.apply(_make_text, axis=1)
    return news_df