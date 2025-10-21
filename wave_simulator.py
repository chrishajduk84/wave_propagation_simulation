"""
Main wave propagation simulator module.
Orchestrates the entire FDTD simulation with configurable parameters.
"""

import fdtd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from snow_layers import SnowLayerStack, SnowLayerProperties
from waveform_generator import WaveformGenerator, WaveformConfig, WaveformType
from detector import DetectorArray, DetectorConfig, Detector
from visualizer import Visualizer, VisualizationConfig, PostProcessingVisualizer


@dataclass
class SimulationConfig:
    """Configuration for the wave propagation simulation."""
    # Grid parameters
    grid_size_m: Tuple[float, float, float] = (0.42, 0.35, 0.001)  # (h, w, z) in meters
    wavelength_m: float = 59.96e-3  # 59.96 mm for 5 GHz
    grid_spacing_factor: int = 100  # Grid points per wavelength
    
    # Time parameters
    courant_number: float = 0.7
    total_time_s: float = 1e-6  # Total simulation time
    
    # PML parameters
    pml_thickness_points: int = 60
    
    # Ski parameters
    ski_thickness_m: float = 3.8e-3  # 3.8 mm
    ski_width_m: float = 100e-3     # 100 mm
    ski_permittivity: float = 2.5
    ski_depth_offset: int = 11      # Additional offset from PML
    
    # Physical constants
    c: float = 2.998e8  # Speed of light
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.frequency_hz = self.c / self.wavelength_m
        self.dx = self.wavelength_m / self.grid_spacing_factor
        self.dt = self.courant_number * self.dx / self.c
        self.total_steps = int(self.total_time_s / self.dt)
        
        # Convert grid size to points
        self.grid_h = int(self.grid_size_m[0] / self.dx)
        self.grid_w = int(self.grid_size_m[1] / self.dx)
        self.grid_z = 1  # 2D simulation


class WavePropagationSimulator:
    """
    Main simulator class for wave propagation in snow layers.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.grid = None
        self.snow_layers = None
        self.waveform_generator = None
        self.detectors = None
        self.visualizer = None
        
        # Simulation state
        self.current_step = 0
        self.time_steps = []
        self.is_initialized = False
        
        # Initialize FDTD backend
        fdtd.set_backend("numpy")
    
    def setup_snow_layers(self, layers: List[SnowLayerProperties]):
        """
        Setup snow layer stack.
        
        Args:
            layers: List of snow layer properties
        """
        self.snow_layers = SnowLayerStack(layers, self.config.dx)
        print(f"Snow layers configured:\n{self.snow_layers}")
    
    def setup_waveform(self, waveform_config: WaveformConfig):
        """
        Setup waveform generator.
        
        Args:
            waveform_config: Waveform configuration
        """
        self.waveform_generator = WaveformGenerator(waveform_config)
        self.waveform_array = self.waveform_generator.generate()
        print(f"Waveform configured: {waveform_config.waveform_type.value}, "
              f"{waveform_config.frequency_hz/1e9:.1f} GHz, "
              f"{len(self.waveform_array)} samples")
    
    def setup_detectors(self, detector_configs: List[DetectorConfig]):
        """
        Setup detector array.
        
        Args:
            detector_configs: List of detector configurations
        """
        detectors = [Detector(config, self.config.dx, self.config.dt) 
                    for config in detector_configs]
        self.detectors = DetectorArray(detectors)
        print(f"Detectors configured: {[d.config.name for d in detectors]}")
    
    def setup_visualization(self, vis_config: VisualizationConfig):
        """
        Setup visualization.
        
        Args:
            vis_config: Visualization configuration
        """
        self.visualizer = Visualizer(vis_config, self.config.dx)
        print(f"Visualization configured: field={vis_config.show_field_plot}, "
              f"detector={vis_config.show_detector_plot}")
    
    def initialize_grid(self):
        """Initialize the FDTD grid with all objects."""
        if self.snow_layers is None:
            raise RuntimeError("Snow layers not configured")
        if self.waveform_generator is None:
            raise RuntimeError("Waveform not configured")
        if self.detectors is None:
            raise RuntimeError("Detectors not configured")
        
        # Create FDTD grid
        self.grid = fdtd.Grid(
            shape=(self.config.grid_h, self.config.grid_w, self.config.grid_z),
            grid_spacing=self.config.dx,
            courant_number=self.config.courant_number
        )
        
        # Add PML boundaries
        pml = self.config.pml_thickness_points
        self.grid[0:pml, :, :] = fdtd.PML(name="pml_xlow")
        self.grid[-pml:, :, :] = fdtd.PML(name="pml_xhigh")
        self.grid[:, 0:pml, :] = fdtd.PML(name="pml_ylow")
        self.grid[:, -pml:, :] = fdtd.PML(name="pml_yhigh")
        
        # Calculate ski positions
        ski_depth = self.config.ski_depth_offset + pml
        ski_thickness_points = int(self.config.ski_thickness_m / self.config.dx)
        ski_width_points = int(self.config.ski_width_m / self.config.dx)
        
        # Position two skis side by side
        space_between_skis = (self.config.grid_w - 2 * ski_width_points) // 3
        left_ski_y = space_between_skis
        right_ski_y = 2 * space_between_skis + ski_width_points
        
        # Add ski layers
        ski_permittivity = self.config.ski_permittivity ** 2
        self.grid[ski_depth:ski_depth+ski_thickness_points, 
                 left_ski_y:left_ski_y+ski_width_points, 0] = \
            fdtd.Object(permittivity=ski_permittivity, name="ski_layer_1")
        
        self.grid[ski_depth:ski_depth+ski_thickness_points,
                 right_ski_y:right_ski_y+ski_width_points, 0] = \
            fdtd.Object(permittivity=ski_permittivity, name="ski_layer_2")
        
        # Add snow layers
        snow_start = ski_depth + ski_thickness_points
        current_depth = snow_start
        self.layer_boundaries = [snow_start]
        
        for layer in self.snow_layers.layers:
            layer_end = current_depth + layer.thickness_grid_points
            self.grid[current_depth:layer_end, :, 0] = \
                fdtd.Object(permittivity=layer.get_fdtd_permittivity(), name=layer.name)
            self.layer_boundaries.append(layer_end)
            current_depth = layer_end
        
        # Add source
        source_position = ski_depth - 2
        source_y = left_ski_y + ski_width_points // 2
        waveform_fdtd = self.waveform_generator.to_fdtd_format(self.config.dx)
        
        self.grid[source_position, source_y, 0] = \
            fdtd.sources.SoftArbitraryPointSource(name="source", waveform_array=waveform_fdtd)
        
        # Place detectors
        self.detectors.place_all_in_grid(self.grid)
        
        self.is_initialized = True
        print(f"Grid initialized: {self.grid}")
        print(f"Ski positions: left_y={left_ski_y}, right_y={right_ski_y}")
        print(f"Source position: x={source_position}, y={source_y}")
    
    def run_simulation(self, progress_callback: Optional[callable] = None) -> Dict:
        """
        Run the complete simulation.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with simulation results
        """
        if not self.is_initialized:
            raise RuntimeError("Simulator not initialized. Call initialize_grid() first.")
        
        print(f"Starting simulation: {self.config.total_steps} steps")
        print(f"Waveform duration: {len(self.waveform_array)} steps")
        
        # Main simulation loop
        for step in range(self.config.total_steps):
            self.current_step = step
            
            # Advance FDTD simulation
            self.grid.step()
            
            # Record time
            current_time = step * self.config.dt
            self.time_steps.append(current_time)
            
            # Collect detector data
            self.detectors.collect_all_data(current_time)
            
            # Update visualization
            if self.visualizer and self.visualizer.should_update(step):
                is_transmitting = step < len(self.waveform_array)
                
                self.visualizer.update_field_plot(
                    self.grid, self.layer_boundaries, is_transmitting
                )
                
                detector_data_dict = self.detectors.get_all_data()
                self.visualizer.update_detector_plot(
                    detector_data_dict, step, self.time_steps
                )
                
                self.visualizer.pause_for_update()
            
            # Progress callback
            if progress_callback and step % 100 == 0:
                progress_callback(step, self.config.total_steps)
            
            # Simple progress print
            if step % (self.config.total_steps // 20) == 0:
                print(f"Progress: {step}/{self.config.total_steps} "
                      f"({100*step/self.config.total_steps:.1f}%)")
        
        # Finalize visualization
        if self.visualizer:
            self.visualizer.finalize()
        
        print("Simulation completed!")
        
        # Return results
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Get simulation results in a structured format.
        
        Returns:
            Dictionary with all simulation results
        """
        results = {
            'simulation_config': {
                'grid_size_points': (self.config.grid_h, self.config.grid_w, self.config.grid_z),
                'grid_size_m': self.config.grid_size_m,
                'dx': self.config.dx,
                'dt': self.config.dt,
                'total_steps': self.config.total_steps,
                'total_time_s': self.config.total_time_s,
                'frequency_hz': self.config.frequency_hz
            },
            'time_array': np.array(self.time_steps),
            'waveform_metadata': self.waveform_generator.get_metadata() if self.waveform_generator else None,
            'snow_layers': self.snow_layers.to_training_data() if self.snow_layers else None,
            'detector_data': self.detectors.to_training_data() if self.detectors else None
        }
        
        return results
    
    def save_results(self, filename: str):
        """Save simulation results to file."""
        import pickle
        results = self.get_results()
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
    
    @classmethod
    def create_default_5ghz_simulation(cls, show_visualization: bool = True,
                                     animation_scale_factor: int = 10) -> 'WavePropagationSimulator':
        """
        Create a default 5 GHz simulation setup.
        
        Args:
            show_visualization: Whether to show real-time visualization
            animation_scale_factor: Visualization update frequency
            
        Returns:
            Configured WavePropagationSimulator
        """
        # Create simulation config
        sim_config = SimulationConfig()
        simulator = cls(sim_config)
        
        # Setup default snow layers
        default_layers = [
            SnowLayerProperties("fresh_snow", 0.0672, 1.4),    # 67.2mm
            SnowLayerProperties("aged_snow", 0.072, 1.7),      # 72mm
            SnowLayerProperties("wet_snow", 0.072, 2.2),       # 72mm
            SnowLayerProperties("compacted_snow", 0.072, 2.8), # 72mm
            SnowLayerProperties("ice_layer", 0.0864, 3.2)      # 86.4mm
        ]
        simulator.setup_snow_layers(default_layers)
        
        # Setup default 5 GHz sine wave
        waveform_config = WaveformConfig(
            waveform_type=WaveformType.SINE,
            frequency_hz=sim_config.frequency_hz,
            duration_s=sim_config.total_time_s / 300,  # Shorter duration like original
            sample_rate_hz=1/sim_config.dt,
            amplitude=1.0
        )
        simulator.setup_waveform(waveform_config)
        
        # Setup ski detector
        ski_depth_m = (sim_config.ski_depth_offset + sim_config.pml_thickness_points) * sim_config.dx
        detector_configs = [
            DetectorConfig(
                name="ski_detector",
                position_m=(ski_depth_m, 0.235, 0),  # Approximate position
                size_m=(0.001, sim_config.ski_width_m/5, 0.001),  # Small but positive dimensions
                detector_type="line"
            )
        ]
        simulator.setup_detectors(detector_configs)
        
        # Setup visualization
        if show_visualization:
            vis_config = VisualizationConfig(
                show_field_plot=True,
                show_detector_plot=True,
                animation_scale_factor=animation_scale_factor
            )
            simulator.setup_visualization(vis_config)
        
        return simulator


# LSTM Training Integration Points
"""
LSTM Training and Inference Integration Guidelines:

1. DATA COLLECTION FOR TRAINING:
   - Use WavePropagationSimulator.get_results() to collect training data
   - Run multiple simulations with varied snow layer configurations
   - Collect detector data for each configuration

Example training data collection:
```python
training_data = []
for config in snow_layer_configurations:
    simulator = WavePropagationSimulator.create_default_5ghz_simulation(show_visualization=False)
    simulator.setup_snow_layers(config)
    simulator.initialize_grid()
    results = simulator.run_simulation()
    training_data.append(results)
```

2. FEATURE EXTRACTION:
   - Input features: Detector time series data (Ex, Ey, Ez components)
   - Target labels: Snow layer properties (thickness, permittivity for each layer)
   - Consider using IQ components for frequency domain features
   - Window the time series data for LSTM input sequences

3. LSTM MODEL ARCHITECTURE:
   - Input: Detector time series (shape: [batch_size, sequence_length, num_features])
   - Output: Snow layer parameters (shape: [batch_size, num_layers * 2])  # thickness + permittivity per layer
   - Consider multi-output model for each layer separately

4. INFERENCE INTEGRATION:
   - Create a LSTMPredictor class that takes DetectorData and returns predicted snow layers
   - Use the same preprocessing pipeline as training
   - Convert predictions back to SnowLayerProperties objects

Example inference integration:
```python
class LSTMPredictor:
    def __init__(self, model_path: str):
        self.model = load_lstm_model(model_path)
    
    def predict_snow_layers(self, detector_data: DetectorData) -> List[SnowLayerProperties]:
        # Extract features from detector data
        features = self._extract_features(detector_data)
        
        # Make prediction
        predictions = self.model.predict(features)
        
        # Convert to SnowLayerProperties
        return self._predictions_to_snow_layers(predictions)
    
    def _extract_features(self, detector_data: DetectorData):
        # Extract time series features for LSTM input
        # This should match the training preprocessing
        pass
```

5. INTEGRATION WITH SIMULATOR:
   - Add method to simulator to accept LSTM predictions
   - Use predicted layers for forward simulation validation
   - Create feedback loop for iterative refinement

6. VALIDATION:
   - Compare LSTM predictions with ground truth snow layers
   - Run forward simulation with predicted layers and compare detector outputs
   - Use metrics like MSE for layer properties and correlation for detector signals
"""