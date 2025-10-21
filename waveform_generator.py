"""
Waveform generation module for wave propagation simulation.
Supports various signal types with IQ sampling capabilities.
"""

import numpy as np
from typing import Union, Tuple, Dict, Callable
from enum import Enum
from dataclasses import dataclass
import warnings


class WaveformType(Enum):
    """Supported waveform types."""
    SINE = "sine"
    SAWTOOTH = "sawtooth"
    SHEPARD_TONE = "shepard_tone"
    CHIRP = "chirp"
    PULSE = "pulse"
    CUSTOM = "custom"


@dataclass
class WaveformConfig:
    """Configuration for waveform generation."""
    waveform_type: WaveformType
    frequency_hz: float
    duration_s: float
    sample_rate_hz: float
    amplitude: float = 1.0
    
    # Specific parameters for different waveform types
    frequency_end_hz: float = None  # For chirp
    pulse_width_s: float = None     # For pulse
    shepard_components: int = 8     # For Shepard tone
    custom_func: Callable = None    # For custom waveforms
    
    def __post_init__(self):
        """Validate configuration."""
        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be positive")
        if self.duration_s <= 0:
            raise ValueError("Duration must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")
        if self.sample_rate_hz < 2 * self.frequency_hz:
            warnings.warn("Sample rate should be at least 2x frequency (Nyquist criterion)")


class IQConverter:
    """Utility class for IQ (In-phase/Quadrature) conversions."""
    
    @staticmethod
    def real_to_iq(signal: np.ndarray, carrier_freq: float, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert real signal to IQ components.
        
        Args:
            signal: Real-valued input signal
            carrier_freq: Carrier frequency for demodulation
            sample_rate: Sample rate of the signal
            
        Returns:
            Tuple of (I, Q) components
        """
        t = np.arange(len(signal)) / sample_rate
        carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
        carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
        
        # Demodulate to baseband
        i_component = signal * carrier_cos
        q_component = -signal * carrier_sin
        
        # Apply low-pass filter (simple moving average for now)
        # In practice, you'd use a proper filter
        window_size = max(1, int(sample_rate / (4 * carrier_freq)))
        i_filtered = np.convolve(i_component, np.ones(window_size)/window_size, mode='same')
        q_filtered = np.convolve(q_component, np.ones(window_size)/window_size, mode='same')
        
        return i_filtered, q_filtered
    
    @staticmethod
    def iq_to_real(i_component: np.ndarray, q_component: np.ndarray, 
                   carrier_freq: float, sample_rate: float) -> np.ndarray:
        """
        Convert IQ components back to real signal.
        
        Args:
            i_component: In-phase component
            q_component: Quadrature component
            carrier_freq: Carrier frequency for modulation
            sample_rate: Sample rate
            
        Returns:
            Real-valued signal
        """
        t = np.arange(len(i_component)) / sample_rate
        carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
        carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
        
        return i_component * carrier_cos - q_component * carrier_sin
    
    @staticmethod
    def iq_to_magnitude_phase(i_component: np.ndarray, q_component: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert IQ to magnitude and phase."""
        magnitude = np.sqrt(i_component**2 + q_component**2)
        phase = np.arctan2(q_component, i_component)
        return magnitude, phase


class WaveformGenerator:
    """
    Generates various types of waveforms for simulation input.
    """
    
    def __init__(self, config: WaveformConfig):
        """
        Initialize waveform generator.
        
        Args:
            config: Waveform configuration
        """
        self.config = config
        self.time_array = None
        self.waveform = None
        self.iq_data = None
        
    def generate(self) -> np.ndarray:
        """
        Generate the waveform based on configuration.
        
        Returns:
            Generated waveform array
        """
        # Create time array
        num_samples = int(self.config.duration_s * self.config.sample_rate_hz)
        self.time_array = np.arange(num_samples) / self.config.sample_rate_hz
        
        # Generate waveform based on type
        if self.config.waveform_type == WaveformType.SINE:
            self.waveform = self._generate_sine()
        elif self.config.waveform_type == WaveformType.SAWTOOTH:
            self.waveform = self._generate_sawtooth()
        elif self.config.waveform_type == WaveformType.SHEPARD_TONE:
            self.waveform = self._generate_shepard_tone()
        elif self.config.waveform_type == WaveformType.CHIRP:
            self.waveform = self._generate_chirp()
        elif self.config.waveform_type == WaveformType.PULSE:
            self.waveform = self._generate_pulse()
        elif self.config.waveform_type == WaveformType.CUSTOM:
            self.waveform = self._generate_custom()
        else:
            raise ValueError(f"Unsupported waveform type: {self.config.waveform_type}")
        
        # Apply amplitude scaling
        self.waveform *= self.config.amplitude
        
        return self.waveform
    
    def get_iq_components(self, carrier_freq: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IQ components of the generated waveform.
        
        Args:
            carrier_freq: Carrier frequency for IQ conversion (defaults to config frequency)
            
        Returns:
            Tuple of (I, Q) components
        """
        if self.waveform is None:
            self.generate()
        
        if carrier_freq is None:
            carrier_freq = self.config.frequency_hz
        
        i_comp, q_comp = IQConverter.real_to_iq(
            self.waveform, carrier_freq, self.config.sample_rate_hz
        )
        
        self.iq_data = (i_comp, q_comp)
        return i_comp, q_comp
    
    def _generate_sine(self) -> np.ndarray:
        """Generate sine wave."""
        return np.sin(2 * np.pi * self.config.frequency_hz * self.time_array)
    
    def _generate_sawtooth(self) -> np.ndarray:
        """Generate sawtooth wave."""
        period = 1.0 / self.config.frequency_hz
        return 2 * (self.time_array % period) / period - 1
    
    def _generate_shepard_tone(self) -> np.ndarray:
        """Generate Shepard tone (illusion of continuously ascending pitch)."""
        signal = np.zeros_like(self.time_array)
        base_freq = self.config.frequency_hz
        
        for i in range(self.config.shepard_components):
            freq = base_freq * (2 ** i)
            # Gaussian envelope to fade higher frequencies
            envelope = np.exp(-0.5 * ((i - self.config.shepard_components/2) / (self.config.shepard_components/4)) ** 2)
            signal += envelope * np.sin(2 * np.pi * freq * self.time_array)
        
        return signal / np.max(np.abs(signal))  # Normalize
    
    def _generate_chirp(self) -> np.ndarray:
        """Generate linear chirp (frequency sweep)."""
        if self.config.frequency_end_hz is None:
            raise ValueError("frequency_end_hz must be specified for chirp waveform")
        
        # Linear chirp
        f0 = self.config.frequency_hz
        f1 = self.config.frequency_end_hz
        t = self.time_array
        T = self.config.duration_s
        
        return np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T)))
    
    def _generate_pulse(self) -> np.ndarray:
        """Generate pulse waveform."""
        if self.config.pulse_width_s is None:
            raise ValueError("pulse_width_s must be specified for pulse waveform")
        
        signal = np.zeros_like(self.time_array)
        pulse_samples = int(self.config.pulse_width_s * self.config.sample_rate_hz)
        
        # Create pulse envelope
        if pulse_samples > 0:
            signal[:pulse_samples] = np.sin(2 * np.pi * self.config.frequency_hz * self.time_array[:pulse_samples])
        
        return signal
    
    def _generate_custom(self) -> np.ndarray:
        """Generate custom waveform using provided function."""
        if self.config.custom_func is None:
            raise ValueError("custom_func must be specified for custom waveform")
        
        return self.config.custom_func(self.time_array, self.config)
    
    def to_fdtd_format(self, dx: float) -> np.ndarray:
        """
        Convert waveform to FDTD-compatible format.
        
        Args:
            dx: Grid spacing for FDTD simulation
            
        Returns:
            Waveform array scaled for FDTD
        """
        if self.waveform is None:
            self.generate()
        
        # Scale by grid spacing (as in original code)
        return self.waveform * dx
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about the generated waveform.
        
        Returns:
            Dictionary with waveform metadata for analysis/training
        """
        metadata = {
            'waveform_type': self.config.waveform_type.value,
            'frequency_hz': self.config.frequency_hz,
            'duration_s': self.config.duration_s,
            'sample_rate_hz': self.config.sample_rate_hz,
            'amplitude': self.config.amplitude,
            'num_samples': len(self.waveform) if self.waveform is not None else 0
        }
        
        if self.config.frequency_end_hz is not None:
            metadata['frequency_end_hz'] = self.config.frequency_end_hz
        if self.config.pulse_width_s is not None:
            metadata['pulse_width_s'] = self.config.pulse_width_s
        if hasattr(self.config, 'shepard_components'):
            metadata['shepard_components'] = self.config.shepard_components
            
        return metadata
    
    @classmethod
    def create_default_5ghz(cls, duration_s: float = 1e-6, sample_rate_hz: float = 100e9) -> 'WaveformGenerator':
        """Create a default 5 GHz sine wave generator."""
        config = WaveformConfig(
            waveform_type=WaveformType.SINE,
            frequency_hz=5e9,  # 5 GHz
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            amplitude=1.0
        )
        return cls(config)