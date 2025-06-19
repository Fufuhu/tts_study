from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download
import sounddevice as sd


def main():
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    #モデルの重みが格納されているpath
    # model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
    # config_file = "jvnv-F1-jp/config.json"
    # style_file = "jvnv-F1-jp/style_vectors.npy"

    model_file = "amitaro/amitaro.safetensors"
    config_file = "amitaro/config.json"
    style_file = "amitaro/style_vectors.npy"

    assets_root = Path("model_assets")

    for file in [model_file, config_file, style_file]:
        print(file)
        # hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="model_assets")
        hf_hub_download("litagin/sbv2_amitaro", file, local_dir="model_assets")

    #モデルインスタンスの作成
    model = TTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device="cpu",
    )

    #「こんにちは」と発話
    text = """
インターネットで手に入るもの、入らないもの
情報の入手と分析にかかるコストのギャップ
    """
    sr, audio = model.infer(text=text)
    #再生
    sd.play(audio, sr)
    sd.wait()

if __name__ == "__main__":
    main()
