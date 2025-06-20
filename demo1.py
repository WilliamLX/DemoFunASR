from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
import time
import csv



# audio path
audios_path = "/Users/william/Work/VoiceData/data"

# model preparation
model_dir = "iic/SenseVoiceSmall"
transcript_model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="mps",
)

vad_model = AutoModel(model="fsmn-vad")

# Initialize empty list to store audio file paths
audio_files = []

# Walk through the directory tree starting from audios_path
for root, dirs, files in os.walk(audios_path):
    # Iterate through all files in current directory
    for file in files:
        # Check if file has a supported audio extension (case-insensitive)
        if file.lower().endswith(('.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg')):
            # Add full path of audio file to the list
            audio_files.append(os.path.join(root, file))


# Iterate through all audio files in the list
for audio_path in audio_files:
    voice_length = vad_model.generate(input=audio_path)
    voice_length = voice_length[0]["value"]
    voice_length = sum(v[1] - v[0] for v in voice_length)/1000
    print("voice_length:",voice_length,'s')
    if voice_length < 30:
        continue

    start_time = time.time()
    print(audio_path)
    # English transcription
    # Generate transcription for each audio file
    res = transcript_model.generate(
        input=audio_path,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        ban_emo_unk=False,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Create the output directory if it doesn't exist
    output_dir = "/Users/william/Work/VoiceData/data/text"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the output file path
    output_file_path = os.path.join(output_dir, f"{base_filename}.txt")
    
    # Write the transcribed text to the file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Transcription saved to: {output_file_path}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    
    # Create logs directory if it doesn't exist
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Write to CSV file
    csv_path = os.path.join(logs_dir, "process_times.csv")
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([elapsed_time])
    





