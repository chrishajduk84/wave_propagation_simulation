"""
Detector module for wave propagation simulation.
Handles detector placement, data collection, and signal processing.
"""

import numpy as np
import fdtd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for a detector."""
    name: str
    position_m: Tuple[float, float, float]  # (x, y, z) position in meters
    size_m: Tuple[float, float, float]      # (x, y, z) size in meters
    detector_type: str = "line"             # "line", "point", or "block"
    
    def __post_init__(self):
        """Validate configuration."""
        if any(p < 0 for p in self.position_m):
            raise ValueError("Position coordinates must be non-negative")
        if any(s <= 0 for s in self.size_m):
            raise ValueError("Size dimensions must be positive")
        if self.detector_type not in ["line", "point", "block"]:
            raise ValueError(f"Invalid detector type: {self.detector_type}. Must be 'line', 'point', or 'block'")


class DetectorData:
    """Container for detector measurement data."""
    
    def __init__(self, detector_name: str, sample_rate: float):
        """
        Initialize detector data container.
        
        Args:
            detector_name: Name of the detector
            sample_rate: Sampling rate of the measurements
        """
        self.detector_name = detector_name
        self.sample_rate = sample_rate
        self.time_steps = []
        
        # Electric field components
        self.e_field_x = []
        self.e_field_y = []
        self.e_field_z = []
        
        # Magnetic field components
        self.h_field_x = []
        self.h_field_y = []
        self.h_field_z = []
        
        # Derived quantities
        self.power = []
        self.magnitude = []
        
    def add_measurement(self, time_step: float, e_field: np.ndarray, h_field: np.ndarray):
        """
        Add a measurement at a specific time step.
        
        Args:
            time_step: Time of measurement
            e_field: Electric field vector [Ex, Ey, Ez]
            h_field: Magnetic field vector [Hx, Hy, Hz]
        """
        self.time_steps.append(time_step)
        
        # Store field components
        self.e_field_x.append(e_field[0])
        self.e_field_y.append(e_field[1])
        self.e_field_z.append(e_field[2])
        
        self.h_field_x.append(h_field[0])
        self.h_field_y.append(h_field[1])
        self.h_field_z.append(h_field[2])
        
        # Calculate derived quantities
        e_magnitude = np.sqrt(np.sum(e_field**2))
        self.magnitude.append(e_magnitude)
        
        # Power (simplified - proportional to |E|^2)
        power = np.sum(e_field**2)
        self.power.append(power)
    
    def get_time_array(self) -> np.ndarray:
        """Get time array in seconds."""
        return np.array(self.time_steps)
    
    def get_e_field_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get electric field components as numpy arrays."""
        return (np.array(self.e_field_x), 
                np.array(self.e_field_y), 
                np.array(self.e_field_z))
    
    def get_h_field_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get magnetic field components as numpy arrays."""
        return (np.array(self.h_field_x), 
                np.array(self.h_field_y), 
                np.array(self.h_field_z))
    
    def get_magnitude(self) -> np.ndarray:
        """Get electric field magnitude."""
        return np.array(self.magnitude)
    
    def get_power(self) -> np.ndarray:
        """Get power measurements."""
        return np.array(self.power)
    
    def get_iq_components(self, component: str = 'z', carrier_freq: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IQ components for a specific field component.
        
        Args:
            component: Field component ('x', 'y', 'z', or 'magnitude')
            carrier_freq: Carrier frequency for IQ conversion
            
        Returns:
            Tuple of (I, Q) components
        """
        # Import here to avoid circular imports
        from waveform_generator import IQConverter
        
        if component == 'x':
            signal = np.array(self.e_field_x)
        elif component == 'y':
            signal = np.array(self.e_field_y)
        elif component == 'z':
            signal = np.array(self.e_field_z)
        elif component == 'magnitude':
            signal = self.get_magnitude()
        else:
            raise ValueError(f"Invalid component: {component}")
        
        if carrier_freq is None:
            # Estimate carrier frequency from signal
            carrier_freq = self._estimate_carrier_frequency(signal)
        
        return IQConverter.real_to_iq(signal, carrier_freq, self.sample_rate)
    
    def _estimate_carrier_frequency(self, signal: np.ndarray) -> float:
        """Estimate carrier frequency from signal using FFT."""
        if len(signal) == 0:
            return 1.0  # Default fallback
        
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        
        # Find peak frequency
        magnitude = np.abs(fft)
        peak_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1  # Exclude DC
        
        return abs(freqs[peak_idx])
    
    def to_training_data(self) -> Dict:
        """
        Convert detector data to format suitable for LSTM training.
        
        Returns:
            Dictionary with detector measurements for ML training
        """
        ex, ey, ez = self.get_e_field_components()
        hx, hy, hz = self.get_h_field_components()
        
        return {
            'detector_name': self.detector_name,
            'time_array': self.get_time_array(),
            'sample_rate': self.sample_rate,
            'e_field': {
                'x': ex,
                'y': ey, 
                'z': ez,
                'magnitude': self.get_magnitude()
            },
            'h_field': {
                'x': hx,
                'y': hy,
                'z': hz
            },
            'power': self.get_power(),
            'num_samples': len(self.time_steps)
        }


class Detector:
    """
    Represents a detector in the FDTD simulation.
    """
    
    def __init__(self, config: DetectorConfig, dx: float, dt: float):
        """
        Initialize detector.
        
        Args:
            config: Detector configuration
            dx: Grid spacing in meters
            dt: Time step in seconds
        """
        self.config = config
        self.dx = dx
        self.dt = dt
        self.data = DetectorData(config.name, 1/dt)
        self.fdtd_detector = None
        self._grid_position = None
        self._grid_size = None
        
    def get_grid_position(self) -> Tuple[int, int, int]:
        """Get detector position in grid coordinates."""
        if self._grid_position is None:
            self._grid_position = tuple(int(pos / self.dx) for pos in self.config.position_m)
        return self._grid_position
    
    def get_grid_size(self) -> Tuple[int, int, int]:
        """Get detector size in grid points."""
        if self._grid_size is None:
            self._grid_size = tuple(max(1, int(size / self.dx)) for size in self.config.size_m)
        return self._grid_size
    
    def place_in_grid(self, grid: fdtd.Grid) -> Union[fdtd.detectors.LineDetector, fdtd.detectors.BlockDetector]:
        """
        Place detector in FDTD grid.
        
        Args:
            grid: FDTD grid object
            
        Returns:
            FDTD detector object (LineDetector or BlockDetector)
        """
        x, y, z = self.get_grid_position()
        sx, sy, sz = self.get_grid_size()
        
        if self.config.detector_type == "line":
            # Create line detector - choose the dimension with the largest size
            if sy > max(sx, sz):  # Line along y-axis
                self.fdtd_detector = fdtd.detectors.LineDetector(name=self.config.name)
                grid[x, y:y+sy, z] = self.fdtd_detector
            elif sx > max(sy, sz):  # Line along x-axis
                self.fdtd_detector = fdtd.detectors.LineDetector(name=self.config.name)
                grid[x:x+sx, y, z] = self.fdtd_detector
            else:  # Single point as line detector (default for small detectors)
                self.fdtd_detector = fdtd.detectors.LineDetector(name=self.config.name)
                grid[x, y, z] = self.fdtd_detector
        elif self.config.detector_type == "point":
            # Use line detector for single point (no dedicated point detector)
            self.fdtd_detector = fdtd.detectors.LineDetector(name=self.config.name)
            grid[x, y, z] = self.fdtd_detector
        elif self.config.detector_type == "block":
            # Use block detector for 2D/3D regions
            self.fdtd_detector = fdtd.detectors.BlockDetector(name=self.config.name)
            grid[x:x+sx, y:y+sy, z:z+sz] = self.fdtd_detector
        else:
            raise ValueError(f"Unsupported detector type: {self.config.detector_type}")
        
        return self.fdtd_detector
    
    def collect_data(self, current_time: float):
        """
        Collect data from the FDTD detector.
        
        Args:
            current_time: Current simulation time
        """
        if self.fdtd_detector is None:
            raise RuntimeError("Detector not placed in grid yet")
        
        detector_values = self.fdtd_detector.detector_values()
        
        # Extract field data - get the latest time step (last element)
        if 'E' in detector_values and len(detector_values['E']) > 0:
            # Get the most recent time step data
            latest_e_data = detector_values['E'][-1]  # Last time step
            
            if len(latest_e_data) > 0:
                # Average over all detector points at this time step
                e_field = np.mean(latest_e_data, axis=0)
            else:
                e_field = np.zeros(3)
        else:
            e_field = np.zeros(3)
        
        if 'H' in detector_values and len(detector_values['H']) > 0:
            # Get the most recent time step data
            latest_h_data = detector_values['H'][-1]  # Last time step
            
            if len(latest_h_data) > 0:
                # Average over all detector points at this time step
                h_field = np.mean(latest_h_data, axis=0)
            else:
                h_field = np.zeros(3)
        else:
            h_field = np.zeros(3)
        
        # Ensure we have 3-component vectors
        if len(e_field) < 3:
            e_field = np.pad(e_field, (0, 3-len(e_field)), 'constant')
        if len(h_field) < 3:
            h_field = np.pad(h_field, (0, 3-len(h_field)), 'constant')
        
        self.data.add_measurement(current_time, e_field[:3], h_field[:3])
    
    def get_data(self) -> DetectorData:
        """Get collected detector data."""
        return self.data
    
    def __repr__(self) -> str:
        return (f"Detector(name='{self.config.name}', "
                f"pos={self.config.position_m}, "
                f"size={self.config.size_m}, "
                f"type='{self.config.detector_type}')")


class DetectorArray:
    """
    Manages multiple detectors in a simulation.
    """
    
    def __init__(self, detectors: List[Detector]):
        """
        Initialize detector array.
        
        Args:
            detectors: List of detector objects
        """
        self.detectors = detectors
        self._detector_map = {det.config.name: det for det in detectors}
    
    def place_all_in_grid(self, grid: fdtd.Grid):
        """Place all detectors in the FDTD grid."""
        for detector in self.detectors:
            detector.place_in_grid(grid)
    
    def collect_all_data(self, current_time: float):
        """Collect data from all detectors."""
        for detector in self.detectors:
            detector.collect_data(current_time)
    
    def get_detector(self, name: str) -> Detector:
        """Get detector by name."""
        if name not in self._detector_map:
            raise ValueError(f"Detector '{name}' not found")
        return self._detector_map[name]
    
    def get_all_data(self) -> Dict[str, DetectorData]:
        """Get data from all detectors."""
        return {name: det.get_data() for name, det in self._detector_map.items()}
    
    def to_training_data(self) -> Dict:
        """
        Convert all detector data to training format.
        
        Returns:
            Dictionary with all detector data for ML training
        """
        training_data = {
            'num_detectors': len(self.detectors),
            'detector_names': list(self._detector_map.keys()),
            'detectors': {}
        }
        
        for name, detector in self._detector_map.items():
            training_data['detectors'][name] = detector.get_data().to_training_data()
        
        return training_data
    
    @classmethod
    def create_ski_detector(cls, ski_position: Tuple[float, float, float], 
                           ski_width_m: float, dx: float, dt: float) -> 'DetectorArray':
        """
        Create a detector array for ski layer measurements.
        
        Args:
            ski_position: Position of ski layer (x, y, z) in meters
            ski_width_m: Width of ski in meters
            dx: Grid spacing
            dt: Time step
            
        Returns:
            DetectorArray with ski detector
        """
        detector_width_m = ski_width_m / 5  # 1/5 of ski width
        detector_config = DetectorConfig(
            name="ski_detector",
            position_m=(ski_position[0], 
                       ski_position[1] + ski_width_m/2 - detector_width_m/2, 
                       ski_position[2]),
            size_m=(0, detector_width_m, 0),
            detector_type="line"
        )
        
        detector = Detector(detector_config, dx, dt)
        return cls([detector])