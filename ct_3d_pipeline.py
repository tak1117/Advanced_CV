import vtk
import os
import csv
import time

INPUT_DIR = "8bit" 
OUTPUT_DIR = "output"

def process_and_save(threshold, reduction, image_data, output_dir, log_list, iteration_info):
    print(f"\n[{iteration_info}] Processing: Threshold={threshold}, Reduction={reduction*100:.0f}%")
    iteration_start_time = time.time()
    
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(image_data)
    marching_cubes.SetValue(0, threshold)
    marching_cubes.ComputeNormalsOn()
    marching_cubes.ComputeGradientsOn()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(marching_cubes.GetOutputPort())

    clean_poly_data = vtk.vtkCleanPolyData()
    clean_poly_data.SetInputConnection(triangle_filter.GetOutputPort())
    
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(clean_poly_data.GetOutputPort())
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()
    decimate.Update()
    
    poly_data = decimate.GetOutput()
    
    if poly_data.GetNumberOfPolys() == 0:
        print("  -> No polygons generated with this combination. Skipping.")
        return

    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(poly_data)
    
    num_polygons = poly_data.GetNumberOfPolys()
    surface_area = mass_properties.GetSurfaceArea()
    volume = mass_properties.GetVolume()
    bounds = poly_data.GetBounds()
    bounding_box_size = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    center_of_mass_filter = vtk.vtkCenterOfMass()
    center_of_mass_filter.SetInputData(poly_data)
    center_of_mass_filter.SetUseScalarsAsWeights(False)
    center_of_mass_filter.Update()
    center = center_of_mass_filter.GetCenter()

    file_base_name = f"model_th{threshold}_red{int(reduction*100)}"

    stl_filepath = os.path.join(output_dir, f"{file_base_name}.stl")
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(stl_filepath)
    stl_writer.SetInputData(poly_data)
    stl_writer.SetFileTypeToBinary()
    stl_writer.Write()
    print(f"  -> Saved STL: {stl_filepath}")

    property = vtk.vtkProperty()
    property.SetColor(0.7, 0.7, 0.7)
    actor = vtk.vtkActor()
    actor.SetMapper(vtk.vtkPolyDataMapper())
    actor.GetMapper().SetInputData(poly_data)
    actor.SetProperty(property)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.9, 0.9, 0.9)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)
    render_window.SetOffScreenRendering(1)
    
    renderer.ResetCamera()
    render_window.Render()

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.Update()

    png_filepath = os.path.join(output_dir, f"{file_base_name}.png")
    png_writer = vtk.vtkPNGWriter()
    png_writer.SetFileName(png_filepath)
    png_writer.SetInputConnection(window_to_image.GetOutputPort())
    png_writer.Write()
    print(f"  -> Saved Image: {png_filepath}")

    log_list.append({
        "Threshold": threshold,
        "Reduction (%)": int(reduction * 100),
        "Polygon Count": num_polygons,
        "Surface Area": f"{surface_area:.2f}",
        "Volume": f"{volume:.2f}",
        "Bounding Box X": f"{bounding_box_size[0]:.2f}",
        "Bounding Box Y": f"{bounding_box_size[1]:.2f}",
        "Bounding Box Z": f"{bounding_box_size[2]:.2f}",
        "Center of Mass X": f"{center[0]:.2f}",
        "Center of Mass Y": f"{center[1]:.2f}",
        "Center of Mass Z": f"{center[2]:.2f}",
        "STL File": f"{file_base_name}.stl",
        "Image File": f"{file_base_name}.png"
    })
    print(f"  -> Iteration finished in {time.time() - iteration_start_time:.2f} seconds.")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: '{os.path.abspath(OUTPUT_DIR)}'")

    print("Reading CT scan files...")
    start_time = time.time()
    try:
        tif_files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff'))])
        if not tif_files:
            print(f"Error: No TIF files found in '{INPUT_DIR}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{INPUT_DIR}'. Please set the correct path.")
        return

    image_append = vtk.vtkImageAppend()
    image_append.SetAppendAxis(2)
    for filepath in tif_files:
        tiff_reader = vtk.vtkTIFFReader()
        tiff_reader.SetFileName(filepath)
        tiff_reader.Update()
        image_append.AddInputData(tiff_reader.GetOutput())
    image_append.Update()
    volume_data = image_append.GetOutput()
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds.")
    print(f"Data value range is: {volume_data.GetScalarRange()}")
    
    results_log = []
    total_iterations = 29
    current_iteration = 0

    print("\n--- Starting Batch 1: Varying Threshold, Reduction=0% ---")
    thresholds1 = range(0, 251, 10)
    reduction1 = 0.0
    for th in thresholds1:
        current_iteration += 1
        iteration_info_str = f"{current_iteration}/{total_iterations}"
        process_and_save(th, reduction1, volume_data, OUTPUT_DIR, results_log, iteration_info_str)

    print("\n--- Starting Batch 2: Threshold=0, Varying Reduction ---")
    threshold2 = 120
    reductions2 = [0.0, 0.5, 0.9, 0.99]
    for red in reductions2:
        current_iteration += 1
        iteration_info_str = f"{current_iteration}/{total_iterations}"
        process_and_save(threshold2, red, volume_data, OUTPUT_DIR, results_log, iteration_info_str)

    if not results_log:
        print("\nNo models were generated. Please check your THRESHOLDS values.")
        return
        
    csv_filepath = os.path.join(OUTPUT_DIR, "comparison_summarynew.csv")
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "Threshold", "Reduction (%)", "Polygon Count", "Surface Area", "Volume",
                "Bounding Box X", "Bounding Box Y", "Bounding Box Z",
                "Center of Mass X", "Center of Mass Y", "Center of Mass Z",
                "STL File", "Image File"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_log)
        print(f"\n✅ All processing complete. Summary saved to: {csv_filepath}")
    except IOError as e:
        print(f"Error: Could not write summary to '{csv_filepath}'. Reason: {e}")

if __name__ == '__main__':
    main()