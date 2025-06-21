from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

def load_bert_japanese_model(model_name: str):
    """
    日本語のBERTモデルとトークナイザーをロードします。
    Args:
        model_name (str): 使用するBERTモデル名
    """
    bert_models.load_model(Languages.JP, model_name)
    bert_models.load_tokenizer(Languages.JP, model_name)
