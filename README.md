# FunASR Demo Repository

This repository contains various demo scripts showcasing FunASR (Fundamental Audio Speech Recognition) capabilities for audio processing and speech recognition tasks.

## Environment Setup

This project uses a dedicated Python virtual environment named `voiceProcess`. 

### Activating the Environment

```bash
source /Users/william/Work/FunASR/funasr/bin/activate
```

When activated, you'll see `(voiceProcess)` in your terminal prompt.

### Environment Details

- **Python Version**: 3.9.6
- **Location**: `/Users/william/Work/VoiceData/voiceProcess`
- **Status**: ✅ **ACTIVE** (voiceProcess environment is currently activated)
- **Key Packages**: 
  - pydub (0.25.1)
  - SpeechRecognition (3.10.0)
  - requests (2.32.4)
  - certifi (2025.6.15)
  - pip (25.1.1)

### Installing FunASR

The FunASR library is required for the main functionality but is not currently installed in the environment. Install it with:

```bash
pip install funasr
```

## Current Status

- **Environment**: ✅ voiceProcess virtual environment activated
- **FunASR**: ❌ Not installed (required for main functionality)
- **Project Management**: 📋 TODO.md created for task tracking
- **Audio Data Path**: `/Users/william/Work/VoiceData/data`
- **Output Path**: `/Users/william/Work/VoiceData/data/text`

## Contents

- **demo1.py** - Basic FunASR demonstration script
- **demo2_understand.py** - Enhanced understanding demo with FunASR
- **voiceActivityDetection.py** - Voice activity detection implementation
- **simple_bark_detection.py** - Simple dog bark detection using audio analysis
- **dog_bark_detection.py** - Advanced dog bark detection with more features
- **ollama_example.py** - Integration example with Ollama for AI processing

## Requirements

- Python 3.x
- FunASR library
- Additional dependencies as specified in each script

## Usage

Each script can be run independently. Check the individual script files for specific usage instructions and requirements.

## Logs

The `logs/` directory contains processing logs and timing data for performance analysis.

## License

This project is open source. Please check individual files for specific licensing information. 