# Audio Processor & Visualizer

Real-time audio signal processing using TTNN for DSP operations.

## Features

- Mel-spectrogram computation
- MFCC extraction
- Beat detection
- Pitch tracking
- Audio effects (reverb, pitch shift, echo, chorus)
- Real-time visualization

## Quick Start

```bash
pip install -r requirements.txt

# Process an audio file
python processor.py examples/sample.wav

# Real-time microphone spectrogram
python visualizer.py --realtime
```

## Files

- `processor.py` - Core audio operations (STFT, mel-spectrogram, MFCC)
- `effects.py` - Audio effects implementation
- `visualizer.py` - Real-time visualization
- `examples/sample.wav` - Example audio (you can use your own files)

## Complete Implementation

See **Lesson 12** for the complete 600+ line implementation including:
- Full AudioProcessor class
- AudioEffects class with reverb, pitch shift, time stretch
- SpectrogramVisualizer with real-time display
- Extensions for audio engineers (VAD, AGC, noise gate, parametric EQ)

## Example Usage

```python
from processor import AudioProcessor
import ttnn

device = ttnn.open_device(device_id=0)
processor = AudioProcessor(device, sample_rate=22050)

# Compute mel-spectrogram
audio = processor.load_audio("your_audio.wav")
mel_spec = processor.compute_mel_spectrogram(audio)

# Detect beats
beats = processor.detect_beats(audio)
print(f"Found {len(beats)} beats")

# Extract pitch
times, pitches = processor.extract_pitch(audio)

ttnn.close_device(device)
```

## Extensions for Audio Engineers

- Voice Activity Detection (VAD)
- Automatic Gain Control (AGC)
- Noise Gate
- Parametric EQ
- VST Plugin Interface

All implementations available in Lesson 12!