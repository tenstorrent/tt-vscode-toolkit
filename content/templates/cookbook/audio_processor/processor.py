"""
Audio Processor using TTNN

Requirements:
    pip install librosa matplotlib

This is a starter template. See Lesson 12 for the complete 400+ line implementation
with full STFT, mel-spectrogram, MFCC, beat detection, and pitch tracking.

Complete implementation includes:
- AudioProcessor class with all DSP operations
- Mel filterbank computation
- FFT-based spectral analysis
- Beat detection with onset strength
- Pitch extraction with YIN algorithm
"""

import ttnn
import torch
import numpy as np
import librosa

class AudioProcessor:
    def __init__(self, device, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128):
        """
        Initialize audio processor.

        Args:
            device: TTNN device handle
            sample_rate: Audio sample rate (Hz)
            n_fft: FFT window size
            hop_length: Samples between frames
            n_mels: Number of mel frequency bins
        """
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        print(f"AudioProcessor initialized: {sample_rate}Hz, {n_fft} FFT, {n_mels} mel bins")
        print("NOTE: See Lesson 12 for complete implementation")

    def load_audio(self, file_path, duration=None):
        """Load audio file."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
        return torch.from_numpy(audio).float()

    def compute_mel_spectrogram(self, audio):
        """
        Compute mel-spectrogram.

        For complete TTNN-accelerated implementation, see Lesson 12.
        This version uses librosa for demonstration.
        """
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        # Using librosa for now - see Lesson 12 for TTNN implementation
        mel_spec = librosa.feature.melspectrogram(
            y=audio.numpy(),
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db


# Example usage
if __name__ == "__main__":
    import os

    # Check for required dependencies
    try:
        import librosa
    except ImportError:
        print("❌ librosa is required. Install with: pip install librosa")
        exit(1)

    device = ttnn.open_device(device_id=0)
    processor = AudioProcessor(device, sample_rate=22050)

    # Load and process audio
    print("\nProcessing audio...")
    audio_file = "examples/sample.wav"  # Use your own file

    try:
        mel_spec = processor.compute_mel_spectrogram(audio_file)
        print(f"\n✅ Mel-spectrogram computed! Shape: {mel_spec.shape}")

        # Visualize with headless environment support
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Check if we're in a headless environment
            if 'DISPLAY' not in os.environ and matplotlib.get_backend() != 'agg':
                print("📊 Headless environment detected, using non-interactive backend...")
                matplotlib.use('Agg')  # Non-interactive backend

            # Create visualization
            plt.figure(figsize=(12, 4))
            plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Spectrogram')
            plt.xlabel('Time (frames)')
            plt.ylabel('Mel Frequency')
            plt.tight_layout()

            # Save or show based on environment
            if 'DISPLAY' not in os.environ or matplotlib.get_backend() == 'agg':
                output_file = 'mel_spectrogram.png'
                plt.savefig(output_file)
                print(f"💾 Visualization saved to {output_file}")
            else:
                plt.show()
                print("🎬 Visualization displayed")

        except ImportError:
            print("\n⚠️  matplotlib not installed. Install with: pip install matplotlib")
            print(f"Mel-spectrogram data is available: shape {mel_spec.shape}")

        print("\n✓ Success! See Lesson 12 for complete TTNN-accelerated implementation.")

    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide your own audio file or use the example from Lesson 12.")
        print("\nExample: processor.compute_mel_spectrogram('path/to/your/audio.wav')")
    except Exception as e:
        print(f"❌ Error: {e}")

    ttnn.close_device(device)
