import os

# audio path
audios_path = "/Users/william/Work/VoiceData/data/audio/"

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

# Print the list of audio files
print(f"Found {len(audio_files)} audio files:")
for audio_file in audio_files:
    print(f"  {audio_file}")
