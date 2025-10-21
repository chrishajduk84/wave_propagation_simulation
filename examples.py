"""
Example usage of the refactored wave propagation simulation.
Demonstrates various simulation configurations and features.
"""

import numpy as np
from wave_simulator import WavePropagationSimulator, SimulationConfig
from snow_layers import SnowLayerProperties, SnowLayerStack
from waveform_generator import WaveformConfig, WaveformType
from detector import DetectorConfig
from visualizer import VisualizationConfig, PostProcessingVisualizer


def example_basic_simulation():
    """Run a basic simulation equivalent to the original stage.py."""
    print("=== Basic 5 GHz Simulation ===")
    
    # Create default simulation
    simulator = WavePropagationSimulator.create_default_5ghz_simulation(
        show_visualization=True,
        animation_scale_factor=10
    )
    
    # Initialize and run
    simulator.initialize_grid()
    results = simulator.run_simulation()
    
    # Save results
    simulator.save_results("basic_simulation_results.pkl")
    
    return results


def example_custom_snow_layers():
    """Example with custom snow layer configuration."""
    print("=== Custom Snow Layers Simulation ===")
    
    # Create simulator with default config
    sim_config = SimulationConfig()
    simulator = WavePropagationSimulator(sim_config)
    
    # Define custom snow layers
    custom_layers = [
        SnowLayerProperties("powder_snow", 0.05, 1.2),      # 50mm powder
        SnowLayerProperties("granular_snow", 0.08, 1.8),    # 80mm granular
        SnowLayerProperties("hard_pack", 0.06, 2.5),        # 60mm hard pack
        SnowLayerProperties("ice_crust", 0.02, 3.5),        # 20mm ice crust
        SnowLayerProperties("base_ice", 0.1, 3.2)           # 100mm base ice
    ]
    
    simulator.setup_snow_layers(custom_layers)
    
    # Use default waveform and detectors
    waveform_config = WaveformConfig(
        waveform_type=WaveformType.SINE,
        frequency_hz=sim_config.frequency_hz,
        duration_s=sim_config.total_time_s / 300,
        sample_rate_hz=1/sim_config.dt,
        amplitude=1.0
    )
    simulator.setup_waveform(waveform_config)
    
    # Setup detector
    ski_depth_m = (sim_config.ski_depth_offset + sim_config.pml_thickness_points) * sim_config.dx
    detector_configs = [
        DetectorConfig(
            name="ski_detector",
            position_m=(ski_depth_m, 0.2, 0),
            size_m=(0, sim_config.ski_width_m/5, 0),
            detector_type="line"
        )
    ]
    simulator.setup_detectors(detector_configs)
    
    # Setup visualization (headless for batch processing)
    vis_config = VisualizationConfig(
        show_field_plot=False,
        show_detector_plot=False
    )
    simulator.setup_visualization(vis_config)
    
    # Run simulation
    simulator.initialize_grid()
    results = simulator.run_simulation()
    
    return results


def example_frequency_sweep():
    """Example with frequency sweep (chirp) waveform."""
    print("=== Frequency Sweep Simulation ===")
    
    sim_config = SimulationConfig()
    simulator = WavePropagationSimulator(sim_config)
    
    # Setup default snow layers
    simulator.setup_snow_layers(SnowLayerStack.create_default_stack(sim_config.dx).layers)
    
    # Frequency sweep from 3 GHz to 8 GHz
    waveform_config = WaveformConfig(
        waveform_type=WaveformType.CHIRP,
        frequency_hz=3e9,        # Start frequency
        frequency_end_hz=8e9,    # End frequency
        duration_s=sim_config.total_time_s / 200,
        sample_rate_hz=1/sim_config.dt,
        amplitude=1.0
    )
    simulator.setup_waveform(waveform_config)
    
    # Setup multiple detectors
    ski_depth_m = (sim_config.ski_depth_offset + sim_config.pml_thickness_points) * sim_config.dx
    detector_configs = [
        DetectorConfig(
            name="detector_1",
            position_m=(ski_depth_m, 0.15, 0),
            size_m=(0, 0.02, 0),
            detector_type="line"
        ),
        DetectorConfig(
            name="detector_2", 
            position_m=(ski_depth_m, 0.25, 0),
            size_m=(0, 0.02, 0),
            detector_type="line"
        )
    ]
    simulator.setup_detectors(detector_configs)
    
    # Visualization off for speed
    vis_config = VisualizationConfig(show_field_plot=False, show_detector_plot=False)
    simulator.setup_visualization(vis_config)
    
    # Run simulation
    simulator.initialize_grid()
    results = simulator.run_simulation()
    
    # Post-processing visualization
    post_viz = PostProcessingVisualizer(sim_config.dx)
    detector_data_dict = simulator.detectors.get_all_data()
    post_viz.plot_detector_summary(detector_data_dict, "frequency_sweep_summary.png")
    
    return results


def example_shepard_tone():
    """Example with Shepard tone (psychoacoustic illusion) waveform."""
    print("=== Shepard Tone Simulation ===")
    
    sim_config = SimulationConfig()
    simulator = WavePropagationSimulator(sim_config)
    
    # Setup snow layers
    simulator.setup_snow_layers(SnowLayerProperties.create_default_stack(sim_config.dx).layers)
    
    # Shepard tone waveform
    waveform_config = WaveformConfig(
        waveform_type=WaveformType.SHEPARD_TONE,
        frequency_hz=sim_config.frequency_hz,
        duration_s=sim_config.total_time_s / 150,
        sample_rate_hz=1/sim_config.dt,
        amplitude=1.0,
        shepard_components=6
    )
    simulator.setup_waveform(waveform_config)
    
    # Setup detector
    ski_depth_m = (sim_config.ski_depth_offset + sim_config.pml_thickness_points) * sim_config.dx
    detector_configs = [
        DetectorConfig(
            name="ski_detector",
            position_m=(ski_depth_m, 0.2, 0),
            size_m=(0, sim_config.ski_width_m/5, 0),
            detector_type="line"
        )
    ]
    simulator.setup_detectors(detector_configs)
    
    # Show only detector plot
    vis_config = VisualizationConfig(
        show_field_plot=False,
        show_detector_plot=True,
        animation_scale_factor=5
    )
    simulator.setup_visualization(vis_config)
    
    # Run simulation
    simulator.initialize_grid()
    results = simulator.run_simulation()
    
    return results


def example_batch_processing():
    """Example of batch processing multiple configurations for LSTM training data."""
    print("=== Batch Processing for LSTM Training Data ===")
    
    # Define parameter variations
    layer_configs = [
        # Thin layers
        [
            SnowLayerProperties("fresh", 0.03, 1.3),
            SnowLayerProperties("aged", 0.04, 1.6),
            SnowLayerProperties("wet", 0.05, 2.1),
            SnowLayerProperties("compact", 0.06, 2.7),
            SnowLayerProperties("ice", 0.07, 3.1)
        ],
        # Thick layers
        [
            SnowLayerProperties("fresh", 0.08, 1.5),
            SnowLayerProperties("aged", 0.09, 1.8),
            SnowLayerProperties("wet", 0.1, 2.3),
            SnowLayerProperties("compact", 0.08, 2.9),
            SnowLayerProperties("ice", 0.12, 3.3)
        ],
        # Mixed thickness
        [
            SnowLayerProperties("fresh", 0.02, 1.2),
            SnowLayerProperties("aged", 0.15, 1.9),
            SnowLayerProperties("wet", 0.03, 2.4),
            SnowLayerProperties("compact", 0.12, 2.6),
            SnowLayerProperties("ice", 0.05, 3.4)
        ]
    ]
    
    frequency_configs = [3e9, 5e9, 7e9, 10e9]  # Different frequencies
    
    all_results = []
    
    for i, layers in enumerate(layer_configs):
        for j, freq in enumerate(frequency_configs):
            print(f"Running configuration {i+1}-{j+1}: {len(layers)} layers, {freq/1e9:.1f} GHz")
            
            # Create simulation
            sim_config = SimulationConfig()
            sim_config.frequency_hz = freq
            sim_config.wavelength_m = sim_config.c / freq
            sim_config.dx = sim_config.wavelength_m / sim_config.grid_spacing_factor
            sim_config.dt = sim_config.courant_number * sim_config.dx / sim_config.c
            sim_config.total_steps = int(sim_config.total_time_s / sim_config.dt)
            
            simulator = WavePropagationSimulator(sim_config)
            
            # Setup layers
            simulator.setup_snow_layers(layers)
            
            # Setup waveform
            waveform_config = WaveformConfig(
                waveform_type=WaveformType.SINE,
                frequency_hz=freq,
                duration_s=sim_config.total_time_s / 300,
                sample_rate_hz=1/sim_config.dt,
                amplitude=1.0
            )
            simulator.setup_waveform(waveform_config)
            
            # Setup detector
            ski_depth_m = (sim_config.ski_depth_offset + sim_config.pml_thickness_points) * sim_config.dx
            detector_configs = [
                DetectorConfig(
                    name="ski_detector",
                    position_m=(ski_depth_m, 0.2, 0),
                    size_m=(0, sim_config.ski_width_m/5, 0),
                    detector_type="line"
                )
            ]
            simulator.setup_detectors(detector_configs)
            
            # No visualization for batch processing
            vis_config = VisualizationConfig(show_field_plot=False, show_detector_plot=False)
            simulator.setup_visualization(vis_config)
            
            # Run simulation
            simulator.initialize_grid()
            results = simulator.run_simulation()
            
            # Add metadata for training
            results['config_id'] = f"{i+1}-{j+1}"
            results['layer_config_index'] = i
            results['frequency_config_index'] = j
            
            all_results.append(results)
    
    # Save all results for LSTM training
    import pickle
    with open("lstm_training_data.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    print(f"Batch processing complete: {len(all_results)} configurations processed")
    return all_results


def example_iq_analysis():
    """Example demonstrating IQ analysis capabilities."""
    print("=== IQ Analysis Example ===")
    
    # Run a basic simulation
    simulator = WavePropagationSimulator.create_default_5ghz_simulation(show_visualization=False)
    simulator.initialize_grid()
    results = simulator.run_simulation()
    
    # Get detector data
    detector_data_dict = simulator.detectors.get_all_data()
    ski_detector_data = detector_data_dict['ski_detector']
    
    # Perform IQ analysis
    post_viz = PostProcessingVisualizer(simulator.config.dx)
    post_viz.plot_iq_analysis(
        ski_detector_data, 
        carrier_freq=simulator.config.frequency_hz,
        component='z',  # Analyze z-component
        save_path="iq_analysis.png"
    )
    
    # Get IQ components for further processing
    i_comp, q_comp = ski_detector_data.get_iq_components('z', simulator.config.frequency_hz)
    print(f"IQ data extracted: I shape={i_comp.shape}, Q shape={q_comp.shape}")
    
    return {'iq_i': i_comp, 'iq_q': q_comp, 'detector_data': ski_detector_data}


if __name__ == "__main__":
    # Run examples
    print("Wave Propagation Simulation Examples")
    print("====================================")
    
    # Choose which example to run
    example_choice = input("""
Choose an example to run:
1. Basic simulation (equivalent to original stage.py)
2. Custom snow layers
3. Frequency sweep (chirp)
4. Shepard tone waveform
5. Batch processing for LSTM training
6. IQ analysis demonstration
7. Run all examples
Enter choice (1-7): """)
    
    if example_choice == "1":
        example_basic_simulation()
    elif example_choice == "2":
        example_custom_snow_layers()
    elif example_choice == "3":
        example_frequency_sweep()
    elif example_choice == "4":
        example_shepard_tone()
    elif example_choice == "5":
        example_batch_processing()
    elif example_choice == "6":
        example_iq_analysis()
    elif example_choice == "7":
        # Run all examples (skip visualization for speed)
        print("Running all examples...")
        example_basic_simulation()
        example_custom_snow_layers()
        example_frequency_sweep()
        example_shepard_tone()
        example_iq_analysis()
        # Skip batch processing as it takes a long time
        print("All examples completed!")
    else:
        print("Invalid choice. Running basic simulation...")
        example_basic_simulation()