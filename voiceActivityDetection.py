from funasr import AutoModel
import os

def get_total_voice_length(data_dir):
    """
    Calculates the total detected voice activity length (in seconds) for all audio files in the given directory.

    Args:
        data_dir (str): Path to the directory containing audio files.

    Returns:
        float: Total voice activity length in seconds.
    """
    model = AutoModel(model="fsmn-vad")
    total_length = 0.0

    # Supported audio extensions
    audio_exts = ('.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg')

    # Walk through the directory and process each audio file
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(audio_exts):
                wav_file = os.path.join(root, file)
                res = model.generate(input=wav_file)
                voice_segments = res[0]["value"]
                for v in voice_segments:
                    total_length += (v[1] - v[0])

    return total_length / 1000  # Convert ms to seconds

# Example usage:
# data_dir = "/Users/william/Work/VoiceData/data/audio/"
# total_seconds = get_total_voice_length(data_dir)
# print(f"Total voice activity length: {total_seconds:.2f} seconds")
