# YouTube動画からの話者分離・文字起こしツール

このプロジェクトは、YouTube動画をダウンロードし、`pyannote.audio`による話者ダイアライゼーションと`Whisper`による音声認識を組み合わせて、話者ごとの文字起こし（WebVTT形式）を生成するPythonスクリプトです。

## 概要

このツールは、YouTube動画から音声を抽出し、高度な話者分離技術と最先端の音声認識モデルを組み合わせることで、動画内の複数の話者の発言を正確に文字起こしし、話者ごとに整理された字幕ファイル（WebVTT形式）を自動生成します。研究、コンテンツ分析、アクセシビリティ向上など、様々な用途で活用できます。

## 機能

*   指定されたYouTube動画のダウンロード
*   動画からの音声抽出（モノラル16kHz WAV形式、ノイズリダクション適用）
*   `pyannote.audio` v3.1 を使用した話者ダイアライゼーション
*   `OpenAI Whisper` を使用した話者ごとの音声認識（文字起こし）
*   話者ラベル付きのWebVTT字幕ファイルの生成

## セットアップ

### 1. Python環境の準備

Python 3.8以上を推奨します。仮想環境の利用を強く推奨します。

```bash
python -m venv venv
source venv/Scripts/activate  # Windowsの場合
# source venv/bin/activate    # macOS/Linuxの場合
```

### 2. 依存関係のインストール

`requirements.txt`に記載されているライブラリをインストールします。

```bash
pip install -r requirements.txt
```

**注意:** `torch`のインストールは、お使いの環境（CPUのみか、CUDA対応GPUがあるか）によって異なります。PyTorchの公式サイト ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) を参照し、適切なコマンドでインストールしてください。


### 3. Hugging Faceアクセストークンの設定

`pyannote.audio`モデルを使用するには、Hugging Faceのアクセストークンが必要です。以下のいずれかの方法で設定してください。

*   **推奨: 環境変数として設定**
    ```bash
    export HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"
    ```
    (Windowsの場合、コマンドプロンプトでは `set HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"`、PowerShellでは `$env:HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"`)
*   **`config.py`を直接編集** (非推奨: セキュリティリスクがあります)
    `config.py`ファイルを開き、`HF_TOKEN`変数を直接編集します。

    ```python
    HF_TOKEN = "hf_YOUR_HUGGING_FACE_TOKEN"
    ```

## 使い方

### スクリプトの実行

`main.py`スクリプトを実行します。必要な引数を指定してください。

```bash
python main.py --video_url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --output_name "my_video_output"
```

**利用可能な引数:**

*   `--video_url` (必須): 処理するYouTube動画のURL。
*   `--output_name` (オプション, デフォルト: `output_video`): 出力ファイルの名前（拡張子なし）。
*   `--cookies_file` (オプション, デフォルト: `None`): `yt-dlp`で使用するクッキーファイルのパス。プライベート動画や年齢制限のある動画のダウンロードに必要となる場合があります。
*   `--whisper_model` (オプション, デフォルト: `turbo`): 使用するWhisperモデル名。利用可能なモデル: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"。
*   `--pyannote_model` (オプション, デフォルト: `pyannote/speaker-diarization-3.1`): 使用する`pyannote.audio`モデル名。
*   `--min_speakers` (オプション, デフォルト: `0`): 最小話者数。0の場合、自動検出。
*   `--max_speakers` (オプション, デフォルト: `0`): 最大話者数。0の場合、自動検出。
*   `--output_dir` (オプション, デフォルト: `output_transcription`): 出力ファイルを保存するディレクトリ。

### Jupyter Notebookでの実行

`pyannotate-sd_whisper.ipynb` Notebookは、`main.py`スクリプトの関数を呼び出す簡単な実行例として機能します。Notebookを開き、セルを順に実行してください。

## 出力

スクリプトの実行後、指定された`output_dir`（デフォルトは`output_transcription`）内に以下のファイルが生成されます。

*   `<output_name>.mp4`: ダウンロードされた動画ファイル。
*   `<output_name>.wav`: 抽出された音声ファイル。
*   `<output_name>.vtt`: 話者ラベル付きのWebVTT字幕ファイル。

## クレジット

*   **pyannote.audio**: 話者ダイアライゼーション
*   **OpenAI Whisper**: 音声認識
*   **yt-dlp**: YouTube動画ダウンロード
*   **ffmpeg-python**: 音声抽出

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については、`LICENSE` ファイルを参照してください。
