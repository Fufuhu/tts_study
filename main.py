from pathlib import Path
from exporter import export_to_mp3
from model_manager import ModelManager
from bert_loader import load_bert_japanese_model

BERT_MODEL_NAME = "ku-nlp/deberta-v2-large-japanese-char-wwm"

# 定数の定義
MODEL_FILE = "amitaro/amitaro.safetensors"
CONFIG_FILE = "amitaro/config.json"
STYLE_FILE = "amitaro/style_vectors.npy"
ASSETS_ROOT = Path("model_assets")


def main():
    load_bert_japanese_model(BERT_MODEL_NAME)

    manager = ModelManager(MODEL_FILE, CONFIG_FILE, STYLE_FILE, ASSETS_ROOT)
    manager.download_model_files()
    model = manager.create_tts_model(device="cpu")

    #「こんにちは」と発話
    text = """
インターネットで手に入るもの、入らないもの
情報の入手と分析にかかるコストのギャップ
    """
    sr, audio = model.infer(text=text)
    export_to_mp3(audio, sr)

if __name__ == "__main__":
    main()
