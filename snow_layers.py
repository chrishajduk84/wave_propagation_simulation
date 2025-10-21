"""
Snow layer module for wave propagation simulation.
Handles snow layer properties and conversions between physical and grid units.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SnowLayerProperties:
    """Properties of a snow layer."""
    name: str
    thickness_m: float  # Thickness in meters
    permittivity: float  # Relative permittivity (εr)
    
    def __post_init__(self):
        """Validate input parameters."""
        if self.thickness_m <= 0:
            raise ValueError("Thickness must be positive")
        if self.permittivity <= 0:
            raise ValueError("Permittivity must be positive")


class SnowLayer:
    """
    Represents a snow layer with physical properties and grid conversions.
    """
    
    def __init__(self, properties: SnowLayerProperties, dx: float):
        """
        Initialize snow layer.
        
        Args:
            properties: Snow layer properties (name, thickness, permittivity)
            dx: Grid spacing in meters
        """
        self.properties = properties
        self.dx = dx
        self._thickness_grid_points = None
        
    @property
    def thickness_m(self) -> float:
        """Get thickness in meters."""
        return self.properties.thickness_m
    
    @property
    def permittivity(self) -> float:
        """Get relative permittivity."""
        return self.properties.permittivity
    
    @property
    def name(self) -> str:
        """Get layer name."""
        return self.properties.name
    
    @property
    def thickness_grid_points(self) -> int:
        """Get thickness in grid points."""
        if self._thickness_grid_points is None:
            self._thickness_grid_points = int(self.thickness_m / self.dx)
        return self._thickness_grid_points
    
    def get_fdtd_permittivity(self) -> float:
        """Get permittivity value for FDTD (squared for electric field)."""
        return self.permittivity ** 2
    
    def __repr__(self) -> str:
        return (f"SnowLayer(name='{self.name}', "
                f"thickness={self.thickness_m:.3f}m, "
                f"permittivity={self.permittivity:.2f}, "
                f"grid_points={self.thickness_grid_points})")


class SnowLayerStack:
    """
    Manages a stack of snow layers for the simulation.
    """
    
    def __init__(self, layers: List[SnowLayerProperties], dx: float):
        """
        Initialize snow layer stack.
        
        Args:
            layers: List of snow layer properties
            dx: Grid spacing in meters
        """
        self.dx = dx
        self.layers = [SnowLayer(props, dx) for props in layers]
        self._calculate_boundaries()
    
    def _calculate_boundaries(self):
        """Calculate grid boundaries for each layer."""
        self.boundaries = []
        cumulative_thickness = 0
        
        for layer in self.layers:
            start = cumulative_thickness
            end = start + layer.thickness_grid_points
            self.boundaries.append((start, end))
            cumulative_thickness = end
    
    def get_boundaries(self) -> List[Tuple[int, int]]:
        """Get grid boundaries for each layer as (start, end) tuples."""
        return self.boundaries.copy()
    
    def get_total_thickness_m(self) -> float:
        """Get total stack thickness in meters."""
        return sum(layer.thickness_m for layer in self.layers)
    
    def get_total_thickness_grid_points(self) -> int:
        """Get total stack thickness in grid points."""
        return sum(layer.thickness_grid_points for layer in self.layers)
    
    def get_layer_by_name(self, name: str) -> SnowLayer:
        """Get a layer by its name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"Layer '{name}' not found")
    
    def to_training_data(self) -> Dict:
        """
        Convert to format suitable for LSTM training.
        
        Returns:
            Dictionary with layer properties for ML training
        """
        return {
            'layer_count': len(self.layers),
            'thicknesses_m': [layer.thickness_m for layer in self.layers],
            'permittivities': [layer.permittivity for layer in self.layers],
            'layer_names': [layer.name for layer in self.layers],
            'total_thickness_m': self.get_total_thickness_m()
        }
    
    @classmethod
    def create_default_stack(cls, dx: float) -> 'SnowLayerStack':
        """Create a default snow layer stack for testing."""
        default_layers = [
            SnowLayerProperties("fresh_snow", 0.0672, 1.4),    # 67.2mm
            SnowLayerProperties("aged_snow", 0.072, 1.7),      # 72mm
            SnowLayerProperties("wet_snow", 0.072, 2.2),       # 72mm
            SnowLayerProperties("compacted_snow", 0.072, 2.8), # 72mm
            SnowLayerProperties("ice_layer", 0.0864, 3.2)      # 86.4mm
        ]
        return cls(default_layers, dx)
    
    def __repr__(self) -> str:
        layer_info = "\n  ".join(str(layer) for layer in self.layers)
        return f"SnowLayerStack(\n  {layer_info}\n  Total: {self.get_total_thickness_m():.3f}m)"