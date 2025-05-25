import os

# Hugging Face Token (環境変数から読み込むことを推奨)
# 例: export HF_TOKEN="hf_YOUR_TOKEN_HERE"
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_DEFAULT_HF_TOKEN_IF_NOT_SET") # デフォルト値は開発用。本番では設定しないこと。

# pyannote.audio model
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# pyannote.audio setting
MIN_SPEAKERS = 0 # 最小話者数 (設定しない場合は0)
MAX_SPEAKERS = 0 # 最大話者数 (設定しない場合は0)

# Whisper model
# 利用可能なモデル: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"
WHISPER_MODEL = "turbo"

# 出力ディレクトリ
OUTPUT_DIR = "output_transcription"
