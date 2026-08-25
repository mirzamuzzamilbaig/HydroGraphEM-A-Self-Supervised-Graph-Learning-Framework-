import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def create_study_area():
    print("Combining img4.png and img5.png side-by-side into prism/study_area.png...")
    # Load images
    img_basin = Image.open("img4.png") # DEM Map
    img_delta = Image.open("img5.png") # Satellite Delta zoom
    
    # Scale delta image to match basin height
    h_target = img_basin.height
    w_delta_target = int(img_delta.width * (h_target / img_delta.height))
    img_delta_resized = img_delta.resize((w_delta_target, h_target), Image.Resampling.LANCZOS)
    
    # Create combined image with a small white divider gap
    divider_w = 40
    combined_w = img_basin.width + w_delta_target + divider_w
    combined = Image.new("RGB", (combined_w, h_target), "white")
    
    combined.paste(img_basin, (0, 0))
    combined.paste(img_delta_resized, (img_basin.width + divider_w, 0))
    
    # Add panel text labels using ImageDraw
    draw = ImageDraw.Draw(combined)
    
    # Draw simple letter label backdrops (Panel a on the left, Panel b on the right)
    draw.rectangle([20, 20, 150, 150], fill="black")
    draw.rectangle([img_basin.width + divider_w + 20, 20, img_basin.width + divider_w + 150, 150], fill="black")
    
    # Draw label text
    try:
        # Try to draw clean letters
        draw.text((60, 40), "a", fill="white", font_size=80)
        draw.text((img_basin.width + divider_w + 60, 40), "b", fill="white", font_size=80)
    except Exception as e:
        print(f"Drawing custom text failed: {e}. Reverting to standard draw.")
        
    combined.save("prism/study_area.png")
    print("Saved combined study area map to prism/study_area.png successfully!")

def draw_methodology_flowchart():
    print("Drawing methodology flowchart using matplotlib...")
    # Clean, modern publication layout
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # Color palette
    c_teal_fill, c_teal_edge, c_teal_text = "#f0fdfa", "#0d9488", "#0f766e"
    c_blue_fill, c_blue_edge, c_blue_text = "#f0f9ff", "#0284c7", "#1e3a8a"
    c_green_fill, c_green_edge, c_green_text = "#f0fdf4", "#16a34a", "#14532d"
    c_slate_fill, c_slate_edge, c_slate_text = "#f8fafc", "#64748b", "#1e293b"
    
    # Box drawer helper
    def draw_box(x, y, w, h, fill, edge, text, text_color, is_bold=False):
        # Draw shadow
        shadow = patches.FancyBboxPatch((x-w/2+0.05, y-h/2-0.05), w, h, 
                                        boxstyle="round,pad=0.1", 
                                        facecolor="#e2e8f0", edgecolor="none", zorder=1)
        ax.add_patch(shadow)
        
        # Draw main box
        box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=fill, edgecolor=edge, lw=2, zorder=2)
        ax.add_patch(box)
        
        # Draw text
        weight = "bold" if is_bold else "normal"
        ax.text(x, y, text, ha="center", va="center", color=text_color, 
                fontsize=9.5, weight=weight, zorder=3)
                
    # Define exact box widths and heights
    box_w = 3.2
    box_h = 1.3
    
    # Column 1: Inputs (x = 2.0)
    text_1a = "SWAT Physical Simulations\n• Channel Flow Rates\n• Sediment Loading\n• Nutrient (NO3/NH4) Fluxes"
    text_1b = "Multi-Modal Earth Obs.\n• MODIS Soil Moisture\n• MODIS Evapotranspiration\n• Landsat NDVI / NDWI"
    text_1c = "Static Terrain Metrics\n• MERIT High-Res DEM\n• Topographic Wetness Index\n• Catchment Slope & Aspect"
    draw_box(2.0, 8.5, box_w, box_h, c_teal_fill, c_teal_edge, text_1a, c_teal_text)
    draw_box(2.0, 6.0, box_w, box_h, c_teal_fill, c_teal_edge, text_1b, c_teal_text)
    draw_box(2.0, 3.5, box_w, box_h, c_teal_fill, c_teal_edge, text_1c, c_teal_text)
    
    # Column 2: Data & Topology (x = 6.0)
    text_2a = "HydroGraphEM Data Engine\n• Temporal Phase Alignment\n• Robust Z-Score Scaling\n• Graph Sequence Formatting"
    text_2b = "Hierarchical Topology\n• 1857 Catchment HRUs\n• 483 SWAT Subbasins\n• 483 Reach Routing Segments"
    draw_box(6.0, 6.0, box_w, box_h, c_blue_fill, c_blue_edge, text_2a, c_blue_text, is_bold=True)
    draw_box(6.0, 3.5, box_w, box_h, c_blue_fill, c_blue_edge, text_2b, c_blue_text, is_bold=True)
    
    # Column 3: Pretraining & Downstream (x = 10.0)
    text_3a = "Self-Supervised Pretraining\n• Spatial-Temporal Masking\n• Flow State Reconstruction\n• Latent Mass Balance"
    text_3b = "HST-PAGT GNN Encoder\n• Spatial Graph Message Passing\n• Temporal Multi-Head Attention\n• Physics-Aware Embeddings"
    text_3c = "Downstream Adaptations\n• P1: Flood Susceptibility\n• P2: Regional Drought Index\n• P3: Water Quality Yields\n• P4: Extreme Event Classifier"
    draw_box(10.0, 8.5, box_w, box_h, c_slate_fill, c_slate_edge, text_3a, c_slate_text)
    draw_box(10.0, 6.0, box_w, box_h, c_green_fill, c_green_edge, text_3b, c_green_text, is_bold=True)
    draw_box(10.0, 2.5, box_w, 2.0, c_green_fill, c_green_edge, text_3c, c_green_text) # Larger height for task list
    
    # Draw connections (arrows) with precise coordinates
    arrow = dict(arrowstyle="-|>", connectionstyle="arc3", color="#475569", lw=2.5, mutation_scale=15, zorder=4)
    
    # Column 1 to Column 2 (points directly to left edge of Box 2A: x=4.4, y=6.0)
    # Box 1A right edge: x=3.6, y=8.5
    # Box 1B right edge: x=3.6, y=6.0
    # Box 1C right edge: x=3.6, y=3.5
    ax.annotate("", xy=(4.3, 6.2), xytext=(3.6, 8.5), arrowprops=arrow)
    ax.annotate("", xy=(4.3, 6.0), xytext=(3.6, 6.0), arrowprops=arrow)
    ax.annotate("", xy=(4.3, 5.8), xytext=(3.6, 3.5), arrowprops=arrow)
    
    # Column 2 steps (Dataset Engine to Hierarchical Topology)
    # Box 2A bottom edge: x=6.0, y=5.35
    # Box 2B top edge: x=6.0, y=4.15
    ax.annotate("", xy=(6.0, 4.25), xytext=(6.0, 5.25), arrowprops=arrow)
    
    # Column 2 to Column 3 (Hierarchical Topology to HST-PAGT GNN Encoder: x=8.4, y=6.0)
    # Box 2B right edge: x=7.6, y=3.5
    ax.annotate("", xy=(8.3, 5.8), xytext=(7.6, 3.5), arrowprops=arrow)
    
    # Column 3 pretraining path to GNN Encoder (Pretraining to HST-PAGT: x=10.0, y=6.0)
    # Box 3A bottom edge: x=10.0, y=7.85
    # Box 3B top edge: x=10.0, y=6.65
    ax.annotate("", xy=(10.0, 6.75), xytext=(10.0, 7.75), arrowprops=arrow)
    
    # GNN Encoder to Downstream Tasks (HST-PAGT to Downstream: x=10.0, y=2.5)
    # Box 3B bottom edge: x=10.0, y=5.35
    # Box 3C top edge: x=10.0, y=3.5
    ax.annotate("", xy=(10.0, 3.6), xytext=(10.0, 5.25), arrowprops=arrow)
    
    # Add section title labels at the top
    ax.text(2.0, 9.6, "1. INPUT DATA STREAMS", ha="center", va="center", fontsize=11, weight="bold", color="#334155")
    ax.text(6.0, 9.6, "2. GRAPH INTEGRATION", ha="center", va="center", fontsize=11, weight="bold", color="#334155")
    ax.text(10.0, 9.6, "3. PRETRAINING & TRANSFERS", ha="center", va="center", fontsize=11, weight="bold", color="#334155")
    
    plt.tight_layout()
    plt.savefig("prism/methodology_flowchart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved methodology flowchart to prism/methodology_flowchart.png successfully!")

def draw_hierarchical_topology():
    print("Drawing hierarchical topology using matplotlib...")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Shaded Background Swimlanes for levels to structure the diagram
    # x bounds: 0.1 to 9.9
    swimlane_1 = patches.Rectangle((0.1, 7.0), 9.8, 2.2, facecolor="#f4fbf7", edgecolor="#e1f5eb", lw=1, alpha=0.5, zorder=0)
    swimlane_2 = patches.Rectangle((0.1, 4.4), 9.8, 2.3, facecolor="#fffaf0", edgecolor="#ffebd6", lw=1, alpha=0.5, zorder=0)
    swimlane_3 = patches.Rectangle((0.1, 1.8), 9.8, 2.3, facecolor="#f0f8ff", edgecolor="#e0f0ff", lw=1, alpha=0.5, zorder=0)
    swimlane_4 = patches.Rectangle((0.1, 0.2), 9.8, 1.3, facecolor="#fff5f5", edgecolor="#ffe3e3", lw=1, alpha=0.5, zorder=0)
    ax.add_patch(swimlane_1)
    ax.add_patch(swimlane_2)
    ax.add_patch(swimlane_3)
    ax.add_patch(swimlane_4)
    
    # Vertical sidebar separator line
    ax.axvline(x=3.0, ymin=0.02, ymax=0.98, color="#cbd5e1", lw=1.5, ls="--", zorder=1)
    
    # Sidebar Text Labels (x = 0.3 to 2.8) - Completely isolated from shapes and arrows
    ax.text(0.3, 8.1, "Level 1: Hydrological\nResponse Units (HRUs)\n• $N_H = 1857$ units\n• Land cover/soil segments", 
            ha="left", va="center", fontsize=10, weight="bold", color="#16a34a", zorder=3)
            
    ax.text(0.3, 5.5, "Level 2: Delineated\nSubbasins\n• $N_S = 483$ catchments\n• Boundary accumulations", 
            ha="left", va="center", fontsize=10, weight="bold", color="#ea580c", zorder=3)
            
    ax.text(0.3, 2.9, "Level 3: Directed\nReach Channels\n• $N_R = 483$ streams\n• Topological routing path", 
            ha="left", va="center", fontsize=10, weight="bold", color="#0284c7", zorder=3)
            
    ax.text(0.3, 0.8, "Level 4: Outlet Node\n• $N_O = 1$ delta outlet\n• Final drainage sink", 
            ha="left", va="center", fontsize=10, weight="bold", color="#dc2626", zorder=3)
            
    # Node coordinates in diagram space (x = 3.2 to 9.8)
    hru_x = np.linspace(3.5, 9.5, 12)
    subbasin_x = np.linspace(4.0, 9.0, 5)
    reach_x = np.linspace(4.5, 8.5, 4)
    outlet_xy = (9.2, 0.8)
    
    # Draw Nodes
    # HRUs (green circles)
    for x in hru_x:
        ax.add_patch(patches.Circle(xy=(x, 8.1), radius=0.15, facecolor="#dcfce7", edgecolor="#16a34a", lw=2, zorder=5))
        
    # Subbasins (orange pentagons)
    for x in subbasin_x:
        ax.add_patch(patches.RegularPolygon(xy=(x, 5.5), numVertices=5, radius=0.28, facecolor="#ffedd5", edgecolor="#ea580c", lw=2, zorder=5))
        
    # Reaches (blue circles representing route segments)
    for x in reach_x:
        ax.add_patch(patches.Circle(xy=(x, 2.95), radius=0.35, facecolor="#e0f2fe", edgecolor="#0284c7", lw=2, zorder=5))
        
    # Outlet (red triangle)
    ax.add_patch(patches.RegularPolygon(xy=outlet_xy, numVertices=3, radius=0.32, facecolor="#fee2e2", edgecolor="#dc2626", lw=2, zorder=5))
    
    # Connections (Arrows)
    arrow = dict(arrowstyle="-|>", connectionstyle="arc3", color="#94a3b8", lw=1.5, mutation_scale=10, zorder=2)
    
    # 1. HRUs to Subbasins routing mapping
    for x_h in hru_x:
        nearest_sb = subbasin_x[np.argmin(np.abs(subbasin_x - x_h))]
        # arrow starts just below HRU and ends at top of Subbasin
        ax.annotate("", xy=(nearest_sb, 5.85), xytext=(x_h, 7.92), arrowprops=arrow)
        
    # 2. Subbasins to Reaches mapping
    for x_s in subbasin_x:
        nearest_reach = reach_x[np.argmin(np.abs(reach_x - x_s))]
        # arrow starts below Subbasin and ends at top of Reach
        ax.annotate("", xy=(nearest_reach, 3.35), xytext=(x_s, 5.18), arrowprops=arrow)
        
    # 3. Reach-to-Reach routing (heavy blue arrows showing downstream flow direction)
    arrow_route = dict(arrowstyle="-|>", connectionstyle="arc3", color="#0284c7", lw=3.2, mutation_scale=16, zorder=4)
    for i in range(len(reach_x)-1):
        # horizontal flow routing from reach to reach
        ax.annotate("", xy=(reach_x[i+1]-0.4, 2.95), xytext=(reach_x[i]+0.4, 2.95), arrowprops=arrow_route)
        
    # 4. Last reach routing to outlet
    # From last reach at (8.5, 2.95) to outlet at (9.2, 0.8)
    ax.annotate("", xy=(outlet_xy[0]-0.25, outlet_xy[1]+0.2), xytext=(8.5, 2.55), arrowprops=arrow_route)
    
    plt.tight_layout()
    plt.savefig("prism/hierarchical_topology.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved hierarchical topology diagram to prism/hierarchical_topology.png successfully!")

if __name__ == "__main__":
    create_study_area()
    draw_methodology_flowchart()
    draw_hierarchical_topology()
