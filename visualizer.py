"""
Visualization module for wave propagation simulation.
Handles real-time plotting and animation with optional display control.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from detector import DetectorData


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    show_field_plot: bool = True
    show_detector_plot: bool = True
    animation_scale_factor: int = 10
    field_plot_size: Tuple[int, int] = (8, 6)
    detector_plot_size: Tuple[int, int] = (8, 6)
    colorbar_limits: Tuple[float, float] = (0, 0.001)
    colorbar_limits_low: Tuple[float, float] = (0, 0.00001)


class Visualizer:
    """
    Handles visualization of wave propagation simulation.
    """
    
    def __init__(self, config: VisualizationConfig, dx: float):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
            dx: Grid spacing for unit conversions
        """
        self.config = config
        self.dx = dx
        self.fig1 = None
        self.fig2 = None
        self.field_colorbar = None
        self.interactive_mode = False
        
        if self.config.show_field_plot or self.config.show_detector_plot:
            self._setup_plots()
    
    def _setup_plots(self):
        """Setup persistent matplotlib figures."""
        plt.ion()  # Turn on interactive mode
        self.interactive_mode = True
        
        if self.config.show_field_plot:
            self.fig1 = plt.figure(1, figsize=self.config.field_plot_size)
            plt.title('Wave Propagation Field')
        
        if self.config.show_detector_plot:
            self.fig2 = plt.figure(2, figsize=self.config.detector_plot_size)
            plt.title('Detector Readings')
        
        plt.show(block=False)
    
    def update_field_plot(self, grid, layer_boundaries: List[int], 
                         is_transmitting: bool = True):
        """
        Update the field visualization plot.
        
        Args:
            grid: FDTD grid object
            layer_boundaries: List of grid boundaries for snow layers
            is_transmitting: Whether the source is currently transmitting
        """
        if not self.config.show_field_plot or self.fig1 is None:
            return
        
        plt.figure(1)
        #plt.clf()  # Don't clear the figure, since fdtd library handles it (grid.visualize)
        
        # Visualize the field
        grid.visualize(z=0, show=False, animate=True)
        
        # Convert axes to mm units
        x_limits = plt.gca().get_xlim()
        y_limits = plt.gca().get_ylim()
        
        # Get current tick locations
        x_ticks = plt.gca().get_xticks()
        y_ticks = plt.gca().get_yticks()
        
        # Convert grid indices to mm (multiply by dx and convert to mm)
        x_labels = [f'{tick * self.dx * 1000:.1f}' for tick in x_ticks]
        y_labels = [f'{tick * self.dx * 1000:.1f}' for tick in y_ticks]
        
        # Set both ticks and labels properly
        plt.gca().set_xticks(x_ticks)
        plt.gca().set_xticklabels(x_labels)
        plt.gca().set_yticks(y_ticks)
        plt.gca().set_yticklabels(y_labels)
        
        # Restore original limits
        plt.gca().set_xlim(x_limits)
        plt.gca().set_ylim(y_limits)
        
        # Add axis labels
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Wave Propagation')
        
        # Add layer boundary lines
        for boundary in layer_boundaries:
            plt.axhline(y=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        # Set color limits based on transmission state
        if is_transmitting:
            plt.clim(*self.config.colorbar_limits)
        else:
            plt.clim(*self.config.colorbar_limits_low)
        
        plt.colorbar()
        plt.tight_layout()
        plt.draw()
    
    def update_detector_plot(self, detector_data_dict: Dict[str, DetectorData], 
                           current_step: int, time_steps: List[float]):
        """
        Update the detector readings plot.
        
        Args:
            detector_data_dict: Dictionary of detector data
            current_step: Current simulation step
            time_steps: List of time steps
        """
        if not self.config.show_detector_plot or self.fig2 is None:
            return
        
        plt.figure(2)
        plt.clf()  # Clear the figure
        
        # Convert time to nanoseconds for display
        time_ns = np.array(time_steps[:current_step+1]) * 1e9
        
        # Plot data from all detectors
        colors = ['r-', 'g-', 'b-', 'm-', 'c-', 'y-']
        color_idx = 0
        
        for detector_name, detector_data in detector_data_dict.items():
            ex, ey, ez = detector_data.get_e_field_components()

            # Only plot up to current step
            if len(ex) > current_step:
                ex = ex[:current_step+1]
                ey = ey[:current_step+1] 
                ez = ez[:current_step+1]
            
            if len(ex) > 0:
                base_color = colors[color_idx % len(colors)][0]
                plt.plot(time_ns, ex, f'{base_color}-', linewidth=1, 
                        label=f'{detector_name} Ex', alpha=0.7)
                plt.plot(time_ns, ey, f'{base_color}--', linewidth=1, 
                        label=f'{detector_name} Ey', alpha=0.7)
                plt.plot(time_ns, ez, f'{base_color}:', linewidth=2, 
                        label=f'{detector_name} Ez')
                color_idx += 1
        
        plt.xlabel('Time (ns)')
        plt.ylabel('Electric Field (V/m)')
        plt.title('Detector Readings - All Components')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Auto-scale y-axis
        if detector_data_dict:
            all_values = []
            for detector_data in detector_data_dict.values():
                ex, ey, ez = detector_data.get_e_field_components()
                current_length = min(len(ex), current_step+1)
                if current_length > 0:
                    all_values.extend(ex[:current_length])
                    all_values.extend(ey[:current_length])
                    all_values.extend(ez[:current_length])
            
            if all_values:
                max_val = max(abs(min(all_values)), abs(max(all_values)))
                if max_val > 0:
                    plt.ylim(-max_val*1.1, max_val*1.1)
        
        plt.tight_layout()
        plt.draw()
    
    def should_update(self, step: int) -> bool:
        """Check if visualization should update at this step."""
        return step % self.config.animation_scale_factor == 0
    
    def pause_for_update(self):
        """Pause to allow plots to update."""
        if self.interactive_mode:
            plt.pause(0.001)
    
    def finalize(self):
        """Finalize visualization and keep plots open."""
        if self.interactive_mode:
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep plots displayed
    
    def save_field_plot(self, filename: str):
        """Save the current field plot."""
        if self.fig1 is not None:
            self.fig1.savefig(filename, dpi=300, bbox_inches='tight')
    
    def save_detector_plot(self, filename: str):
        """Save the current detector plot."""
        if self.fig2 is not None:
            self.fig2.savefig(filename, dpi=300, bbox_inches='tight')
    
    @classmethod
    def create_headless(cls, dx: float) -> 'Visualizer':
        """Create a visualizer that doesn't show plots (for batch processing)."""
        config = VisualizationConfig(
            show_field_plot=False,
            show_detector_plot=False
        )
        return cls(config, dx)
    
    @classmethod
    def create_field_only(cls, dx: float, animation_scale_factor: int = 10) -> 'Visualizer':
        """Create a visualizer that only shows field plots."""
        config = VisualizationConfig(
            show_field_plot=True,
            show_detector_plot=False,
            animation_scale_factor=animation_scale_factor
        )
        return cls(config, dx)
    
    @classmethod
    def create_detector_only(cls, dx: float, animation_scale_factor: int = 10) -> 'Visualizer':
        """Create a visualizer that only shows detector plots."""
        config = VisualizationConfig(
            show_field_plot=False,
            show_detector_plot=True,
            animation_scale_factor=animation_scale_factor
        )
        return cls(config, dx)
    
    @classmethod
    def create_full(cls, dx: float, animation_scale_factor: int = 10) -> 'Visualizer':
        """Create a visualizer that shows both field and detector plots."""
        config = VisualizationConfig(
            show_field_plot=True,
            show_detector_plot=True,
            animation_scale_factor=animation_scale_factor
        )
        return cls(config, dx)


class PostProcessingVisualizer:
    """
    Creates plots after simulation completion for analysis.
    """
    
    def __init__(self, dx: float):
        """
        Initialize post-processing visualizer.
        
        Args:
            dx: Grid spacing for unit conversions
        """
        self.dx = dx
    
    def plot_detector_summary(self, detector_data_dict: Dict[str, DetectorData], 
                            save_path: Optional[str] = None):
        """
        Create a summary plot of all detector data.
        
        Args:
            detector_data_dict: Dictionary of detector data
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detector Data Summary', fontsize=16)
        
        # Plot 1: All E-field components over time
        ax = axes[0, 0]
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        
        for i, (detector_name, detector_data) in enumerate(detector_data_dict.items()):
            time_ns = detector_data.get_time_array() * 1e9
            ex, ey, ez = detector_data.get_e_field_components()
            
            color = colors[i % len(colors)]
            ax.plot(time_ns, ex, f'{color}-', alpha=0.7, label=f'{detector_name} Ex')
            ax.plot(time_ns, ey, f'{color}--', alpha=0.7, label=f'{detector_name} Ey')
            ax.plot(time_ns, ez, f'{color}:', linewidth=2, label=f'{detector_name} Ez')
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Electric Field (V/m)')
        ax.set_title('Electric Field Components')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Field magnitude
        ax = axes[0, 1]
        for i, (detector_name, detector_data) in enumerate(detector_data_dict.items()):
            time_ns = detector_data.get_time_array() * 1e9
            magnitude = detector_data.get_magnitude()
            
            color = colors[i % len(colors)]
            ax.plot(time_ns, magnitude, f'{color}-', linewidth=2, label=f'{detector_name}')
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('|E| (V/m)')
        ax.set_title('Electric Field Magnitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Power
        ax = axes[1, 0]
        for i, (detector_name, detector_data) in enumerate(detector_data_dict.items()):
            time_ns = detector_data.get_time_array() * 1e9
            power = detector_data.get_power()
            
            color = colors[i % len(colors)]
            ax.plot(time_ns, power, f'{color}-', linewidth=2, label=f'{detector_name}')
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Power (V²/m²)')
        ax.set_title('Power')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Frequency spectrum (FFT of Ez component)
        ax = axes[1, 1]
        for i, (detector_name, detector_data) in enumerate(detector_data_dict.items()):
            ex, ey, ez = detector_data.get_e_field_components()
            
            if len(ez) > 1:
                fft = np.fft.fft(ez)
                freqs = np.fft.fftfreq(len(ez), 1/detector_data.sample_rate)
                
                # Only plot positive frequencies
                pos_freqs = freqs[:len(freqs)//2]
                pos_fft = np.abs(fft[:len(fft)//2])
                
                color = colors[i % len(colors)]
                ax.plot(pos_freqs * 1e-9, pos_fft, f'{color}-', linewidth=2, 
                       label=f'{detector_name}')
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('Frequency Spectrum (Ez)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 20)  # Limit to 20 GHz for visibility
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_iq_analysis(self, detector_data: DetectorData, carrier_freq: float,
                        component: str = 'z', save_path: Optional[str] = None):
        """
        Create IQ analysis plots for a detector.
        
        Args:
            detector_data: Detector data to analyze
            carrier_freq: Carrier frequency for IQ conversion
            component: Field component to analyze ('x', 'y', 'z', or 'magnitude')
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'IQ Analysis - {detector_data.detector_name} ({component} component)', fontsize=16)
        
        # Get IQ components
        i_comp, q_comp = detector_data.get_iq_components(component, carrier_freq)
        time_ns = detector_data.get_time_array() * 1e9
        
        # Plot 1: I and Q components
        ax = axes[0, 0]
        ax.plot(time_ns, i_comp, 'b-', linewidth=1, label='I (In-phase)')
        ax.plot(time_ns, q_comp, 'r-', linewidth=1, label='Q (Quadrature)')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Amplitude')
        ax.set_title('IQ Components')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: IQ constellation
        ax = axes[0, 1]
        ax.scatter(i_comp, q_comp, alpha=0.5, s=1)
        ax.set_xlabel('I (In-phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.set_title('IQ Constellation')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 3: Magnitude and phase
        magnitude, phase = IQConverter.iq_to_magnitude_phase(i_comp, q_comp)
        ax = axes[1, 0]
        ax.plot(time_ns, magnitude, 'g-', linewidth=1, label='Magnitude')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Magnitude')
        ax.set_title('IQ Magnitude')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Phase
        ax = axes[1, 1]
        # Import IQConverter here to avoid circular imports
        from waveform_generator import IQConverter
        magnitude, phase = IQConverter.iq_to_magnitude_phase(i_comp, q_comp)
        ax.plot(time_ns, np.unwrap(phase), 'purple', linewidth=1)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Phase (radians)')
        ax.set_title('IQ Phase (unwrapped)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Note: IQConverter imported locally to avoid circular imports