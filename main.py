import os
import sys
import ffmpeg
import whisper
from pyannote.audio import Audio, Pipeline
import torch
import yt_dlp
import argparse
from datetime import timedelta

# config.pyから設定をインポート
sys.path.append(os.path.dirname(__file__))
from config import HF_TOKEN, PYANNOTE_MODEL, MIN_SPEAKERS, MAX_SPEAKERS, WHISPER_MODEL, OUTPUT_DIR

# ffmpeg.exeのパスを定義
FFMPEG_PATH = r'C:\opt\ffmpeg-2023-01-01-git-62da0b4a74-full_build\bin\ffmpeg.exe'

def download_video_with_audio(video_url: str, output_mp4_path: str, cookies_file: str = None):
    """
    YouTube動画を音声付きMP4ファイルとしてダウンロードします。
    Args:
        video_url (str): YouTube動画のURL。
        output_mp4_path (str): ダウンロードするMP4ファイルの出力パス。
        cookies_file (str, optional): yt-dlpで使用するクッキーファイルのパス。
    Returns:
        str: ダウンロードされたMP4ファイルのパス。
    """
    if not os.path.exists(output_mp4_path):
        print(f"動画ファイルが存在しません: {output_mp4_path}. ダウンロードを開始します...")
        ydl_opts = {
            # より汎用的なフォーマット指定:
            # 1. MP4コンテナで、最良のビデオと最良のオーディオを結合 (存在すれば)
            # 2. それがダメなら、単一ファイルで最良のMP4 (音声付き)
            # 3. それもダメなら、利用可能な最良のフォーマット (yt-dlpが自動でMP4に変換しようと試みる)
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
            'outtmpl': output_mp4_path,
            'noplaylist': True,
            'merge_output_format': 'mp4', # マージする場合の出力形式をmp4に指定
            # 'quiet': True, # デバッグ時はコメントアウトして詳細なログを見る
            # 'verbose': True, # さらに詳細なログ
        }
        if cookies_file and os.path.exists(cookies_file):
            ydl_opts['cookiefile'] = cookies_file
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            print(f"動画ファイルがダウンロードされました: {output_mp4_path}.")
        except Exception as e:
            print(f"動画のダウンロード中にエラーが発生しました: {e}")
            raise
    else:
        print(f"動画ファイルが既に存在します: {output_mp4_path}. ダウンロードをスキップします。")
    return output_mp4_path


def convert_mp4_to_wav(input_mp4_path: str, output_wav_path: str, ffmpeg_path: str):
    """
    MP4ファイルからオーディオを抽出し、モノラル16kHzのWAVファイルとして保存します。
    Args:
        input_mp4_path (str): 入力MP4ファイルのパス。
        output_wav_path (str): 出力WAVファイルのパス。
        ffmpeg_path (str): ffmpeg実行ファイルのパス。
    """
    if os.path.exists(output_wav_path):
        print(f"WAVファイルが既に存在します: {output_wav_path}. 変換をスキップします。")
        return

    print(f"WAVファイルが存在しません: {output_wav_path}. MP4からの変換を開始します...")
    try:
        # MP4ファイルを入力として指定
        # 出力フォーマットがWAVなので、FFmpegは自動的にオーディオストリームを処理します
        (
            ffmpeg
            .input(input_mp4_path)
            .output(output_wav_path, acodec='pcm_s16le', ac=1, ar=16000, format='wav', af='afftdn=nr=12') # モノラル16kHz, ノイズリダクション
            .overwrite_output()
            .run(cmd=ffmpeg_path, quiet=True) # ffmpegのパスを明示的に指定
        )
        print(f"MP4からオーディオが抽出され、{output_wav_path} として保存されました。")
    except ffmpeg.Error as e:
        print(f"WAVへの変換中にFFmpegエラーが発生しました: {e.stderr.decode('utf8')}")
        raise
    except Exception as e:
        print(f"WAVへの変換中にエラーが発生しました: {e}")
        raise

def transcribe_with_diarization(audio_path: str, whisper_model_name: str, pyannote_model_name: str, hf_token: str, min_speakers: int = None, max_speakers: int = None):
    """
    音声ファイルに対して話者ダイアライゼーションと文字起こしを実行します。
    (この関数の内容は変更ありません)
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {DEVICE}")

    try:
        model = whisper.load_model(whisper_model_name, device=DEVICE)
        print(f"Whisperモデル '{whisper_model_name}' が正常にロードされました。")
    except Exception as e:
        print(f"Whisperモデルのロード中にエラーが発生しました: {e}")
        raise

    try:
        pipeline = Pipeline.from_pretrained(pyannote_model_name, use_auth_token=hf_token)
        pipeline.to(torch.device(DEVICE))
        print(f"Pyannoteパイプライン '{pyannote_model_name}' が正常にロードされました。")
    except Exception as e:
        print(f"Pyannoteパイプラインのロード中にエラーが発生しました: {e}")
        print("古いモデルバージョンを試すか、Hugging Faceトークンが正しく設定されているか確認してください。")
        raise

    print(f"'{audio_path}' で話者ダイアライゼーションを実行中...")
    try:
        diarization = pipeline(audio_path, min_speakers=min_speakers if min_speakers > 0 else None, max_speakers=max_speakers if max_speakers > 0 else None)
        print("話者ダイアライゼーションが完了しました。")
    except FileNotFoundError:
        print(f"エラー: 音声ファイル '{audio_path}' が見つかりません。")
        raise
    except Exception as e:
        print(f"ダイアライゼーション中にエラーが発生しました: {e}")
        raise

    audio_processor = Audio(sample_rate=16000, mono=True)
    print(f"音声ファイル '{audio_path}' をロード中...")
    try:
        full_waveform, sample_rate = audio_processor(audio_path)
        print(f"音声がロードされました: shape={full_waveform.shape}, sample_rate={sample_rate}")
    except Exception as e:
        print(f"音声ファイルのロード中にエラーが発生しました: {e}")
        raise

    print("セグメントの文字起こし中...")
    transcribed_segments = []
    if full_waveform is not None and sample_rate is not None:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            try:
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                
                waveform = full_waveform[:, start_sample:end_sample]
                audio_data_np = waveform[0, :].numpy()

                min_samples = int(0.1 * sample_rate)
                if audio_data_np.shape[0] < min_samples:
                    continue

                result = model.transcribe(audio_data_np, language="ja", fp16=torch.cuda.is_available())
                text = result["text"]

                print(f"  [{segment.start:05.3f}s - {segment.end:05.3f}s] {speaker}: {text}")

                transcribed_segments.append({
                    "segment": segment,
                    "speaker": speaker,
                    "text": text
                })

            except Exception as e:
                print(f"  セグメント {segment.start:.3f}s - {segment.end:.3f}s の文字起こし中にエラーが発生しました (話者: {speaker}): {e}")
                continue
    else:
        print("音声ファイルのロードエラーのため、文字起こしをスキップします。")

    print("処理が完了しました。")
    return transcribed_segments

def save_as_webvtt(transcribed_segments: list, output_vtt_path: str):
    """
    文字起こし結果をWebVTT形式でファイルに保存します。
    (この関数の内容は変更ありません)
    """
    print(f"結果を {output_vtt_path} に保存中...")

    def format_time(milliseconds):
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for item in transcribed_segments:
            segment = item["segment"]
            speaker = item["speaker"]
            text = item["text"]

            start_time_ms = int(segment.start * 1000)
            end_time_ms = int(segment.end * 1000)

            start_time_str = format_time(start_time_ms)
            end_time_str = format_time(end_time_ms)

            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"<v {speaker}> {text.strip()}\n")
            f.write("\n")

    print(f"結果が {output_vtt_path} に保存されました。")

def main():
    parser = argparse.ArgumentParser(description="YouTube動画のダウンロード、音声抽出、話者ダイアライゼーション、文字起こしを行います。")
    parser.add_argument("--video_url", type=str, required=True, help="処理するYouTube動画のURL。")
    parser.add_argument("--output_name", type=str, default="output_video", help="出力ファイルの名前（拡張子なし）。")
    parser.add_argument("--cookies_file", type=str, default=None, help="yt-dlpで使用するクッキーファイルのパス。")
    parser.add_argument("--whisper_model", type=str, default=WHISPER_MODEL, help=f"使用するWhisperモデル名 (デフォルト: {WHISPER_MODEL})。")
    parser.add_argument("--pyannote_model", type=str, default=PYANNOTE_MODEL, help=f"使用するpyannote.audioモデル名 (デフォルト: {PYANNOTE_MODEL})。")
    parser.add_argument("--min_speakers", type=int, default=MIN_SPEAKERS, help=f"最小話者数 (デフォルト: {MIN_SPEAKERS})。")
    parser.add_argument("--max_speakers", type=int, default=MAX_SPEAKERS, help=f"最大話者数 (デフォルト: {MAX_SPEAKERS})。")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help=f"出力ファイルを保存するディレクトリ (デフォルト: {OUTPUT_DIR})。")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ファイル名を定義
    mp4_file_name = f"{args.output_name}.mp4" # ダウンロードする音声付きMP4ファイル
    output_wav_file_name = f"{args.output_name}.wav"
    vtt_file_name = f"{args.output_name}.vtt"

    # ファイルパスを定義
    mp4_path = os.path.join(args.output_dir, mp4_file_name)
    audio_wav_path = os.path.join(args.output_dir, output_wav_file_name)
    vtt_path = os.path.join(args.output_dir, vtt_file_name)

    try:
        # 1. 動画を音声付きMP4としてダウンロード
        downloaded_mp4_path = download_video_with_audio(args.video_url, mp4_path, args.cookies_file)

        # 2. ダウンロードされたMP4からオーディオを抽出し、WAVに変換
        convert_mp4_to_wav(downloaded_mp4_path, audio_wav_path, FFMPEG_PATH)

        # 3. 話者ダイアライゼーションと文字起こし
        transcribed_segments = transcribe_with_diarization(
            audio_wav_path,
            args.whisper_model,
            args.pyannote_model,
            HF_TOKEN,
            args.min_speakers,
            args.max_speakers
        )

        # 4. WebVTT形式で保存
        save_as_webvtt(transcribed_segments, vtt_path)

        print("\nすべての処理が正常に完了しました。")
        print(f"出力ファイル: {vtt_path}")

    except Exception as e:
        print(f"\n処理中に致命的なエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()