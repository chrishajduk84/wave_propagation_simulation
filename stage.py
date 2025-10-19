import fdtd
import matplotlib.pyplot as plt
import numpy as np

fdtd.set_backend("numpy")



#########################
# Simulation parameters
#########################
c = 2.998e8
wavelength = 59.96e-3   # 59.96 mm for 5 GHz
frequency = c / wavelength
dx = wavelength / 100   # approximate grid spacing (rule of thumb: 10 points per wavelength)
courant = 0.7
dt = courant * dx / c
total_steps = int(1e-6 / dt)

# Animation speed control
animation_scale_factor = 10  # Higher number = faster animation (skips more steps)

print(f"Total steps: {total_steps}, dx: {dx}, dt: {dt}")
print(f"Animation scale factor: {animation_scale_factor} (showing every {animation_scale_factor} steps)")
t = np.arange(total_steps//300) * dt

h = int(420e-3 / dx)
w = int(350e-3 / dx)
z = 1
print(h,w,z)
input("Continue?")


grid = fdtd.Grid(
    shape = (h, w, z), # 25mm x 15mm x 1 (grid_spacing) --> 2D FDTD
    grid_spacing=dx,
    courant_number=courant
)

# For continuous wave, no envelope
waveform_array = np.sin(2 * np.pi * frequency * t) * dx

pml_offset = 60

# Calculate ski layer thickness in grid points
ski_thickness = int(3.8e-3 / dx)

# Snow layer thicknesses in grid points
fresh_snow_thickness = 112      # to convert to mm multiply by dx*1000
aged_snow_thickness = 120
wet_snow_thickness = 120
compacted_snow_thickness = 120
ice_layer_thickness = 144

# Two ski layers side by side at the same depth - 3.8mm thick each
ski_depth = 11 + pml_offset
ski_width = int(100e-3 / dx)  # 100mm skis

# First ski layer (left side) - 3.8mm thick x 100mm wide
space_between_skis = (w - 2 * ski_width)//3 
left_ski_x = space_between_skis
grid[ski_depth:ski_depth+ski_thickness, left_ski_x:left_ski_x+ski_width, 0] = fdtd.Object(permittivity=2.5**2, name="ski_layer_1")
# Second ski layer (right side) - 3.8mm thick x 100mm wide
right_ski_x = 2*space_between_skis + ski_width
grid[ski_depth:ski_depth+ski_thickness, right_ski_x:right_ski_x+ski_width, 0] = fdtd.Object(permittivity=2.5**2, name="ski_layer_2")

# Snow layers with various permittivities (doubled thickness) - positioned below ski layers
snow_start = ski_depth + ski_thickness

# Calculate snow layer boundaries
fresh_snow_end = snow_start + fresh_snow_thickness
aged_snow_end = fresh_snow_end + aged_snow_thickness  
wet_snow_end = aged_snow_end + wet_snow_thickness
compacted_snow_end = wet_snow_end + compacted_snow_thickness
ice_layer_end = compacted_snow_end + ice_layer_thickness

# Create snow layers using the defined boundaries
grid[snow_start:fresh_snow_end, :, 0] = fdtd.Object(permittivity=1.4**2, name="fresh_snow")
grid[fresh_snow_end:aged_snow_end, :, 0] = fdtd.Object(permittivity=1.7**2, name="aged_snow")
grid[aged_snow_end:wet_snow_end, :, 0] = fdtd.Object(permittivity=2.2**2, name="wet_snow")
grid[wet_snow_end:compacted_snow_end, :, 0] = fdtd.Object(permittivity=2.8**2, name="compacted_snow")
grid[compacted_snow_end:ice_layer_end, :, 0] = fdtd.Object(permittivity=3.2**2, name="ice_layer")

# Add detector in the second ski layer
detector_position = ski_depth# + ski_thickness//2   # Top of second ski layer
detector_width = ski_width//5
grid[detector_position, right_ski_x+ski_width//2-detector_width//2:right_ski_x+ski_width//2+detector_width//2, 0] = fdtd.detectors.LineDetector(name="ski_detector")

#grid[7.5e-6:8.0e-6, 11.8e-6:13.0e-6, 0] = fdtd.LineSource(
#    period = 1550e-9 / (3e8), name="source"
#)
# Update source position to be above the upper ski layer
source_position = ski_depth-2  # Position above the ski layers
grid[source_position, left_ski_x+(ski_width//2), 0] = fdtd.sources.SoftArbitraryPointSource(name="source", waveform_array=waveform_array)

# x boundaries
grid[0:pml_offset, :, :] = fdtd.PML(name="pml_xlow")
grid[-pml_offset:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:pml_offset, :] = fdtd.PML(name="pml_ylow")
grid[:, -pml_offset:, :] = fdtd.PML(name="pml_yhigh")

print(grid)

#grid.run(total_time=1e-12)

# Initialize arrays to store detector data
detector_data_H = [[],[],[]]
detector_data_E = [[],[],[]]
time_steps = []

# Create persistent figures for animation
plt.ion()  # Turn on interactive mode
fig1 = plt.figure(1, figsize=(8, 6))  # Figure 1: Field visualization
fig2 = plt.figure(2, figsize=(8, 6))  # Figure 2: Detector readings
plt.show(block=False)

print("Simulation completed.")
print(f"Running for {total_steps} steps. Waveform source running for {len(waveform_array)} steps.")
for i in range(total_steps):
    grid.step()
    
    # Store detector data at each step
    # Find the ski_detector in the detectors list
    ski_detector = None
    for detector in grid.detectors:
        if detector.name == "ski_detector":
            ski_detector = detector
            break

    if ski_detector is not None:
        # For each time step, compute the average x-vector, y-vector, z-vector
        x = 0
        y = 0
        z = 0
        for data in ski_detector.detector_values()['E'][i]:
            x = x + data[0]
            y = y + data[1]
            z = z + data[2]
        detector_data_E[0].append(x/len(data))
        detector_data_E[1].append(y/len(data))
        detector_data_E[2].append(z/len(data))

    time_steps.append(i * dt)
    
    # Only visualize every nth step based on scale factor
    if i % animation_scale_factor == 0:
        # Update Figure 1: Field visualization
        plt.figure(1)
        grid.visualize(z=0, show=False, animate=True)
        
        # Convert axes to mm units
        # Store original limits to preserve them
        x_limits = plt.gca().get_xlim()
        y_limits = plt.gca().get_ylim()
        
        # Get current tick locations
        x_ticks = plt.gca().get_xticks()
        y_ticks = plt.gca().get_yticks()
        
        # Convert grid indices to mm (multiply by dx and convert to mm)
        x_labels = [f'{tick * dx * 1000:.1f}' for tick in x_ticks]
        y_labels = [f'{tick * dx * 1000:.1f}' for tick in y_ticks]
        
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
        
        # Add light dotted lines to show snow layer boundaries
        # Use the predefined layer boundaries
        layer_boundaries = [
            snow_start,           # Top of fresh snow
            fresh_snow_end,       # Fresh snow -> Aged snow
            aged_snow_end,        # Aged snow -> Wet snow
            wet_snow_end,         # Wet snow -> Compacted snow
            compacted_snow_end,   # Compacted snow -> Ice layer
            ice_layer_end         # Bottom of ice layer
        ]
        
        # Draw horizontal dotted lines at each boundary
        for boundary in layer_boundaries:
            plt.axhline(y=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        # While transmitting, plot at higher clim range, when not transmitting, use lower range
        if len(waveform_array) >= i:
            plt.clim(0,0.001)
        else:
            plt.clim(0,0.00001)
        plt.colorbar()
        plt.tight_layout()
        plt.draw()
        
        # Update Figure 2: Detector readings
        plt.figure(2)
        plt.clf()  # Clear the figure

        # Plot all three vector components
        time_ns = np.array(time_steps[:i+1]) * 1e9
        plt.plot(time_ns, detector_data_E[0][:i+1], 'r-', linewidth=1, label='Ex (x-component)')
        plt.plot(time_ns, detector_data_E[1][:i+1], 'g-', linewidth=1, label='Ey (y-component)')
        plt.plot(time_ns, detector_data_E[2][:i+1], 'b-', linewidth=1, label='Ez (z-component)')
        
        plt.xlabel('Time (ns)')
        plt.ylabel('Electric Field (V/m)')
        plt.title('Detector Reading (Ski Layer) - All Components')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set y-axis limits for consistency considering all components
        if len(detector_data_E[0]) > 0 and len(detector_data_E[1]) > 0 and len(detector_data_E[2]) > 0:
            all_values = detector_data_E[0][:i+1] + detector_data_E[1][:i+1] + detector_data_E[2][:i+1]
            if all_values:
                max_val = max(abs(min(all_values)), abs(max(all_values)))
                if max_val > 0:
                    plt.ylim(-max_val*1.1, max_val*1.1)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to allow plots to update
        print(f"Step {i+1}/{total_steps} complete.", end="\r")

print("Visualization completed.")

# Keep plots open
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep plots displayed