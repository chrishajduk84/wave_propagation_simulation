"""
Refactored version of the original stage.py using the new modular architecture.
This demonstrates how to achieve the same results as the original monolithic code.
"""

from wave_simulator import WavePropagationSimulator
from snow_layers import SnowLayerProperties
from waveform_generator import WaveformConfig, WaveformType
from detector import DetectorConfig
from visualizer import VisualizationConfig


def main():
    """Main function replicating the original stage.py behavior."""
    
    print("Refactored Wave Propagation Simulation")
    print("=====================================")
    
    # Create the simulator with the same parameters as original
    simulator = WavePropagationSimulator.create_default_5ghz_simulation(
        show_visualization=True,
        animation_scale_factor=10
    )
    
    # The default setup already matches the original:
    # - 5 GHz sine wave
    # - Default snow layers (fresh, aged, wet, compacted, ice)
    # - Ski detector in the right ski layer
    # - Real-time visualization
    
    print(f"Grid size: {simulator.config.grid_h} x {simulator.config.grid_w} x {simulator.config.grid_z}")
    print(f"Grid spacing: {simulator.config.dx*1000:.3f} mm")
    print(f"Time step: {simulator.config.dt*1e12:.3f} ps")
    print(f"Total steps: {simulator.config.total_steps}")
    print(f"Frequency: {simulator.config.frequency_hz/1e9:.1f} GHz")
    
    # Show snow layer configuration
    print(f"\nSnow layer configuration:")
    print(simulator.snow_layers)
    
    # Ask user if they want to continue (same as original)
    response = input("\nContinue with simulation? (y/n): ")
    if response.lower() != 'y':
        print("Simulation cancelled.")
        return
    
    # Initialize the simulation
    print("\nInitializing simulation...")
    simulator.initialize_grid()
    
    # Run the simulation
    print("Starting simulation...")
    results = simulator.run_simulation()
    
    # Save results
    simulator.save_results("refactored_stage_results.pkl")
    
    print("\nSimulation completed successfully!")
    print(f"Results saved to: refactored_stage_results.pkl")
    
    # Show summary of results
    detector_data = simulator.detectors.get_all_data()
    ski_detector = detector_data['ski_detector']
    
    print(f"\nDetector Summary:")
    print(f"- Samples collected: {len(ski_detector.time_steps)}")
    print(f"- Sampling rate: {ski_detector.sample_rate/1e9:.1f} GS/s")
    print(f"- Total measurement time: {ski_detector.get_time_array()[-1]*1e9:.1f} ns")
    
    # Show field component statistics
    ex, ey, ez = ski_detector.get_e_field_components()
    print(f"- Ex component: min={ex.min():.2e}, max={ex.max():.2e}, mean={ex.mean():.2e}")
    print(f"- Ey component: min={ey.min():.2e}, max={ey.max():.2e}, mean={ey.mean():.2e}")
    print(f"- Ez component: min={ez.min():.2e}, max={ez.max():.2e}, mean={ez.mean():.2e}")
    
    return results


if __name__ == "__main__":
    main()