# Wave Propagation Simulation - Refactored Architecture

This project provides a modular, object-oriented framework for simulating electromagnetic wave propagation through snow layers using FDTD (Finite-Difference Time-Domain) methods. The architecture is designed to support machine learning applications, specifically LSTM training for snow layer parameter estimation.

## Architecture Overview

The original monolithic `stage.py` has been refactored into several specialized modules:

### Core Modules

1. **`snow_layers.py`** - Snow layer definitions and properties
2. **`waveform_generator.py`** - Signal generation with IQ sampling support
3. **`detector.py`** - Detector placement and data collection
4. **`visualizer.py`** - Real-time and post-processing visualization
5. **`wave_simulator.py`** - Main simulation orchestrator

### Key Features

- **Configurable Snow Layers**: Define thickness and permittivity in meters, automatic grid conversion
- **Multiple Waveform Types**: Sine, sawtooth, chirp, Shepard tone, pulse, and custom waveforms
- **IQ Sampling Support**: Built-in I/Q conversion for both input and output signals
- **Flexible Detector Arrays**: Point and line detectors with automatic data collection
- **Optional Visualization**: Real-time animation or headless operation for batch processing
- **Training Data Export**: Structured data format for LSTM model training

## Quick Start

### Basic Usage (Equivalent to Original)

```python
from wave_simulator import WavePropagationSimulator

# Create and run a basic 5 GHz simulation
simulator = WavePropagationSimulator.create_default_5ghz_simulation()
simulator.initialize_grid()
results = simulator.run_simulation()
```

### Custom Configuration

```python
from wave_simulator import WavePropagationSimulator, SimulationConfig
from snow_layers import SnowLayerProperties
from waveform_generator import WaveformConfig, WaveformType

# Create custom simulation
sim_config = SimulationConfig()
simulator = WavePropagationSimulator(sim_config)

# Define custom snow layers
layers = [
    SnowLayerProperties("powder", 0.05, 1.3),  # 50mm, εr=1.3
    SnowLayerProperties("packed", 0.08, 2.1),  # 80mm, εr=2.1
    SnowLayerProperties("ice", 0.06, 3.2)      # 60mm, εr=3.2
]
simulator.setup_snow_layers(layers)

# Configure frequency sweep
waveform_config = WaveformConfig(
    waveform_type=WaveformType.CHIRP,
    frequency_hz=3e9,
    frequency_end_hz=8e9,
    duration_s=1e-6,
    sample_rate_hz=100e9
)
simulator.setup_waveform(waveform_config)

# Run simulation
simulator.initialize_grid()
results = simulator.run_simulation()
```

## File Structure

```
wave_propagation_simulation/
├── snow_layers.py          # Snow layer definitions
├── waveform_generator.py   # Signal generation
├── detector.py             # Detector management
├── visualizer.py           # Visualization tools
├── wave_simulator.py       # Main simulator class
├── examples.py             # Usage examples
├── stage_refactored.py     # Refactored original
├── stage.py               # Original monolithic code
└── README.md              # This file
```

## LSTM Integration

The architecture includes specific support for LSTM model training and inference:

### Training Data Collection

```python
# Batch processing for training data
configurations = generate_snow_layer_configs()
training_data = []

for config in configurations:
    simulator = WavePropagationSimulator.create_default_5ghz_simulation(show_visualization=False)
    simulator.setup_snow_layers(config)
    simulator.initialize_grid()
    results = simulator.run_simulation()
    training_data.append(results)

# Save training dataset
save_training_data(training_data)
```

### Training Data Format

The simulation results include structured data suitable for ML training:

```python
results = {
    'simulation_config': {...},     # Grid parameters, frequencies, etc.
    'time_array': [...],           # Time steps
    'waveform_metadata': {...},    # Input signal characteristics
    'snow_layers': {               # Ground truth layer properties
        'thicknesses_m': [...],
        'permittivities': [...],
        'layer_names': [...]
    },
    'detector_data': {             # Output measurements
        'detectors': {
            'detector_name': {
                'e_field': {'x': [...], 'y': [...], 'z': [...]},
                'h_field': {'x': [...], 'y': [...], 'z': [...]},
                'time_array': [...],
                'sample_rate': ...,
                'power': [...],
                'magnitude': [...]
            }
        }
    }
}
```

### LSTM Model Integration Points

1. **Feature Extraction**: Use detector time series (Ex, Ey, Ez) as input features
2. **Target Labels**: Snow layer thickness and permittivity values
3. **Preprocessing**: IQ conversion, windowing, normalization
4. **Model Architecture**: LSTM → Dense layers → Snow layer parameters
5. **Inference**: Real-time layer estimation from detector measurements

### Example LSTM Workflow

```python
# 1. Data Preprocessing
def prepare_lstm_data(results_list):
    X = []  # Detector time series
    y = []  # Snow layer parameters
    
    for result in results_list:
        # Extract detector data
        detector_data = result['detector_data']['detectors']['ski_detector']
        features = np.column_stack([
            detector_data['e_field']['x'],
            detector_data['e_field']['y'], 
            detector_data['e_field']['z']
        ])
        
        # Extract target labels
        layers = result['snow_layers']
        targets = np.concatenate([
            layers['thicknesses_m'],
            layers['permittivities']
        ])
        
        X.append(features)
        y.append(targets)
    
    return np.array(X), np.array(y)

# 2. Model Training
X_train, y_train = prepare_lstm_data(training_data)
model = create_lstm_model(input_shape=X_train.shape[1:], output_dim=y_train.shape[1])
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 3. Inference
def predict_snow_layers(detector_data, model):
    features = extract_features(detector_data)
    predictions = model.predict(features)
    return predictions_to_snow_layers(predictions)
```

## Advanced Features

### IQ Signal Processing

```python
# Get IQ components from detector data
detector_data = results['detector_data']['detectors']['ski_detector']
i_comp, q_comp = detector_data.get_iq_components('z', carrier_freq=5e9)

# Analyze magnitude and phase
from waveform_generator import IQConverter
magnitude, phase = IQConverter.iq_to_magnitude_phase(i_comp, q_comp)
```

### Multiple Detector Arrays

```python
from detector import DetectorConfig

detector_configs = [
    DetectorConfig("surface", (0.01, 0.2, 0), (0, 0.02, 0)),
    DetectorConfig("middle", (0.15, 0.2, 0), (0, 0.02, 0)),
    DetectorConfig("bottom", (0.3, 0.2, 0), (0, 0.02, 0))
]
simulator.setup_detectors(detector_configs)
```

### Headless Operation

```python
from visualizer import VisualizationConfig

# Disable visualization for batch processing
vis_config = VisualizationConfig(
    show_field_plot=False,
    show_detector_plot=False
)
simulator.setup_visualization(vis_config)
```

## Examples

See `examples.py` for comprehensive usage examples:

- Basic simulation (equivalent to original)
- Custom snow layer configurations
- Frequency sweep analysis
- Shepard tone waveforms
- Batch processing for LSTM training
- IQ analysis and post-processing

## Migration from Original

To migrate from the original `stage.py`:

1. Replace monolithic configuration with modular setup
2. Use `WavePropagationSimulator.create_default_5ghz_simulation()` for equivalent behavior
3. Access results through structured `results` dictionary
4. Use `PostProcessingVisualizer` for analysis plots

The refactored version maintains full compatibility with the original simulation parameters while providing much greater flexibility and extensibility for machine learning applications.

## Dependencies

- `fdtd`: FDTD simulation framework
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `dataclasses`: Configuration management (Python 3.7+)

## License

[Add your license information here]