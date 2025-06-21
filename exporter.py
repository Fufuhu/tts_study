from pydub import AudioSegment
import tempfile
import os
import soundfile as sf
import logging

def export_to_mp3(audio, sr, output_path="output.mp3"):
    """
    一時WAVファイルを作成し、mp3に変換して保存する。
    Args:
        audio: 音声データ
        sr: サンプリングレート
        output_path: 出力するmp3ファイルのパス
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        temp_wav_path = tmp_wav.name
        sf.write(temp_wav_path, audio, sr)
    try:
        sound = AudioSegment.from_wav(temp_wav_path)
        sound.export(output_path, format="mp3")
    except Exception as e:
        logging.error(f"mp3作成時にエラーが発生しました: {e}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

