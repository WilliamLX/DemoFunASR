from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
import time
import csv
from datetime import datetime


def load_models():
    """Load FunASR models once and return them"""
    print("Loading transcription model...")
    transcript_model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="mps",
    )

    print("Loading VAD model...")
    vad_model = AutoModel(model="fsmn-vad")

    return transcript_model, vad_model


def get_audio_files(audios_path):
    """Get all audio files from the specified directory"""
    audio_files = []
    for root, dirs, files in os.walk(audios_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg')):
                audio_files.append(os.path.join(root, file))
    return audio_files


def process_audio_file(audio_path, filename, transcript_model, vad_model, output_dir, logs_dir):
    """Process a single audio file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(logs_dir, f"process_times_{timestamp}.csv")

    # Check voice activity duration
    voice_length = vad_model.generate(input=audio_path)
    voice_length = voice_length[0]["value"]
    voice_length = sum(v[1] - v[0] for v in voice_length) / 1000
    print("voice_length:", voice_length, 's')

    if voice_length < 10:
        print(f"Skipping {audio_path} - voice length too short")
        log_processing_time(log_path, filename, voice_length, -1)
        return None

    start_time = time.time()
    print(f"Processing: {audio_path}")

    # Generate transcription
    res = transcript_model.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        ban_emo_unk=False,
    )
    text = rich_transcription_postprocess(res[0]["text"])

    # Save transcription
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_file_path = os.path.join(output_dir, f"{base_filename}.txt")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Calculate processing time
    elapsed_time = time.time() - start_time
    print(f"Transcription saved to: {output_file_path}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    log_processing_time(log_path, filename, voice_length, elapsed_time)

    return elapsed_time


def log_processing_time(log_path, filename, voice_length, elapsed_time):
    """Log processing time to a CSV file"""
    with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename, voice_length, elapsed_time])


def main():
    """Main function to orchestrate the audio processing"""
    # Configuration
    audios_path = "/Users/william/Work/VoiceData/data"
    output_dir = "/Users/william/Work/VoiceData/data/text"
    logs_dir = "./logs"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load models once
    print("Loading models...")
    start_time = time.time()
    transcript_model, vad_model = load_models()
    model_load_time = time.time() - start_time
    print(f"Models loaded successfully in {model_load_time:.2f} seconds")

    # Get all audio files
    audio_files = get_audio_files(audios_path)
    print(f"Found {len(audio_files)} audio files to process")

    # Process each audio file
    count = 0

    for audio_path in audio_files:
        count += 1
        filename = os.path.basename(audio_path)
        print(f"Processing file {count}: {filename}")
        print(audio_path)
        process_audio_file(audio_path, filename, transcript_model, vad_model, output_dir, logs_dir)


if __name__ == "__main__":
    main()
