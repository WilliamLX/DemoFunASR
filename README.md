# FunASR Demo Repository

This repository contains various demo scripts showcasing FunASR (Fundamental Audio Speech Recognition) capabilities for audio processing and speech recognition tasks.

## Environment Setup

This project uses a dedicated Python virtual environment named `voiceProcess`.

### Activating the Environment

```bash
source voiceProcess/bin/activate
```

When activated, you'll see `(voiceProcess)` in your terminal prompt.

### Environment Details

- **Python Version**: 3.9.6
- **Location**: `./voiceProcess`

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- **funasr** - Speech recognition and VAD
- **librosa** - Audio analysis (for dog bark detection)
- **numpy** - Numerical computing
- **matplotlib** - Visualization (for bark detection)
- **scipy** - Scientific computing (for bark detection)
- **requests** - HTTP library (for Ollama integration)

## Project Structure

```
DemoFunASR/
├── src/                        # Main source code
│   ├── voice_to_text_processor.py   # Audio transcription processor
│   └── file_utils.py                # Text analysis using Ollama
├── demos/                      # Demo scripts and utilities
│   ├── demo2_understand.py     # Audio file discovery
│   ├── voiceActivityDetection.py  # VAD implementation
│   ├── dog_bark_detection.py   # Advanced bark detection
│   ├── simple_bark_detection.py  # Simple bark detection
│   └── ollama_example.py        # Ollama integration example
├── data/
│   └── text/                   # Output: transcribed text files
├── logs/                       # Processing logs
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Contents

### Main Processor
- **voice_to_text_processor.py** - Batch transcribes audio files using FunASR's SenseVoice model
  - Input: Audio files from `/Users/william/Work/VoiceData/data`
  - Output: Transcribed text to `data/text/`
  - Filters audio with <10 seconds of voice activity

### Utilities
- **file_utils.py** - Analyzes transcribed text using Ollama API
  - Checks for insurance/gas sales related content
  - Categorizes conversations

### Demo Scripts (`later/`)
- **demo2_understand.py** - Audio file discovery utility
- **voiceActivityDetection.py** - Voice activity detection using FunASR fsmn-vad
- **simple_bark_detection.py** - Simple dog bark detection
- **dog_bark_detection.py** - Advanced dog bark detection with MFCC/spectral analysis
- **ollama_example.py** - Ollama API integration example

## Models

The project uses these FunASR models:
- **SenseVoiceSmall** (iic/SenseVoiceSmall) - Speech transcription
- **fsmn-vad** - Voice Activity Detection

## Usage

### Transcribe Audio Files
```bash
python src/voice_to_text_processor.py
```

Or with custom paths:
```bash
AUDIOS_PATH=/path/to/audio OUTPUT_DIR=/path/to/output python src/voice_to_text_processor.py
```

### Analyze Transcribed Text
```bash
python src/file_utils.py
```

### Run Demo Scripts
```bash
python demos/voiceActivityDetection.py
python demos/dog_bark_detection.py
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIOS_PATH` | `/Users/william/Work/VoiceData/data` | Input audio directory |
| `OUTPUT_DIR` | `/Users/william/Work/VoiceData/data/text` | Output text directory |
| `LOGS_DIR` | `./logs` | Logs directory |

## Logs

The `logs/` directory contains processing logs and timing data for performance analysis.

## License

This project is open source. Please check individual files for specific licensing information.
