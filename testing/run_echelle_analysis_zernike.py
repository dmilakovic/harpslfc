#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run Echelle Analysis workflow & generate post-analysis plots.
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import the main analyzer class and plotting functions for analysis
from harps.twodim.analyzer import EchelleAnalyzer # Analyzer class
from harps.twodim.plotting import (plot_coefficient_vs_segment,
                                   plot_adjacent_segment_comparison,
                                   plot_coefficient_heatmap_vs_segment,
                                   plot_coefficient_on_detector) # Analysis plotters


def run_analysis(config):
    """Runs the core analysis and incrementally saves the results FITS file."""
    print("\n--- Phase 1: Running Analysis and Saving FITS ---")
    analyzer = EchelleAnalyzer(
        lfc_filename=config['lfc_filename'], bias_filename=config.get('bias_filename'),
        output_dir=config['output_dir'], clobber_output=config.get('clobber_output', True),
        # Pass analysis params from config dict
        peak_min_distance=config['analysis_params']['peak_min_dist'],
        peak_hard_cut=config.get('hard_flux_cut'),
        peak_threshold_abs=config['analysis_params'].get('peak_threshold_abs'),
        peak_threshold_rel=config['analysis_params'].get('peak_threshold_rel'),
        cluster_eps_x=config['analysis_params']['cluster_eps_x'],
        cluster_eps_y=config['analysis_params']['cluster_eps_y'],
        cluster_min_samples=config['analysis_params']['cluster_min_samp'],
        stamp_half_width=config['analysis_params']['stamp_hw'],
        fit_threshold_snr=config['analysis_params']['fit_snr_thresh'],
        num_segments=config['analysis_params']['num_seg'],
        n_max_zern=config['analysis_params']['n_max_z'],
        r_max_zern=config['analysis_params']['r_max_z']
    )

    # --- Run Workflow Steps ---
    success = False
    if analyzer.load_data(detector=config['detector']):
        if analyzer.find_all_peaks(plot_interactive=config['plot_config'].get('interactive_peaks', False)):
            if analyzer.cluster_orders(plot_interactive=config['plot_config'].get('interactive_clusters', False)):
                # --- Perform the analysis and incremental saving ---
                success = analyzer.analyze_and_update( # Call the new combined method
                    orders_to_process=config['processing_control']['orders_to_run'],
                    image_types_to_process=config['processing_control']['images_to_run'],
                    plot_config=config['plot_config']
                )
                if success: print("--- Analysis and FITS Update Complete ---")
                else: print("--- Analysis finished, but issues occurred during processing/saving. ---")
            else: print("Clustering failed.")
        else: print("Peak finding failed.")
    else: print("Data loading failed.")

    if not success: print("--- Phase 1 Failed ---")
    # Return path regardless of success, as file might have been partially written
    return analyzer.output_fits_path


def run_post_analysis_plotting(results_fits_path, config):
    """Reads the FITS file and generates comparison/detector plots."""
    print("\n--- Phase 2: Generating Post-Analysis Plots ---")
    if not results_fits_path or not Path(results_fits_path).is_file():
         print(f"Error: Results FITS file not found at {results_fits_path}. Cannot generate plots.")
         return

    # Need metadata (indices, shape, params) and the results table
    # Read metadata first using a minimal Analyzer instance or standalone function
    # analyzer_metadata = EchelleAnalyzer(config['lfc_filename']).get_fits_metadata(results_fits_path)
    # It's safer to pass the full config and filename to a standalone reader if we had one
    # For now, we can instantiate Analyzer, call get_fits_metadata and read_results_table
    # using the path.

    # Create a temporary Analyzer instance to read the data and metadata
    temp_analyzer = EchelleAnalyzer(config['lfc_filename']) # Need a path, but this instance won't run analysis
    temp_analyzer.output_fits_path = Path(results_fits_path) # Point it to the results file

    analyzer_metadata = temp_analyzer.get_fits_metadata(results_fits_path)
    results_table = temp_analyzer.read_results_table(results_fits_path)


    if results_table is None or analyzer_metadata is None or not analyzer_metadata.get('zernike_indices'):
        print("Failed to read results table, metadata, or Zernike indices. Cannot generate analysis plots.")
        return

    zernike_indices = analyzer_metadata['zernike_indices']
    img_shape = analyzer_metadata.get('image_shape')
    # Recover detector from saved metadata
    detector_str = analyzer_metadata.get('params',{}).get('detector','unk')

    if len(results_table) < 1: # Need at least one segment for some plots
         print("No segment results found in the FITS file.")
         return

    # --- Config for Analysis Plots ---
    plot_config = config['plot_config']
    save_plots = plot_config.get('save_plots', False)
    plot_format = plot_config.get('plot_format', 'pdf').lower() # Default to pdf
    if plot_format not in ['pdf', 'png']:
        print(f"Warning: Invalid plot_format '{plot_format}'. Defaulting to 'pdf'.")
        plot_format = 'pdf'
    analysis_plot_dir = Path(plot_config.get('plot_dir', config['output_dir'] / "plots"))

    comparison_config = plot_config.get('comparison', {})
    heatmap_config = plot_config.get('heatmap', {})
    detector_map_config = plot_config.get('detector_map', {})

    comparison_plots_enabled = comparison_config.get('enable', False)
    heatmap_plot_enabled = heatmap_config.get('enable', False)
    detector_plots_enabled = detector_map_config.get('enable', True)

    # Define plot directory for analysis plots
    comp_subdir = Path(comparison_config.get('subdir', analysis_plot_dir / "comparison"))
    comp_subdir.mkdir(parents=True, exist_ok=True)
    detector_map_dir = Path(detector_map_config.get('subdir', analysis_plot_dir / "detector_maps"))
    detector_map_dir.mkdir(parents=True, exist_ok=True)


    # Define common filename prefix based on what's in the FITS file
    orders_present = np.unique(results_table['ORDER_NUM'])
    imgtypes_present = np.unique(results_table['IMGTYPE'])
    imgtypes_str = "".join(['A' if i==0 else 'B' for i in sorted(imgtypes_present)])
    order_str = f"O{orders_present[0]}" if len(orders_present)==1 else "Oall"
    file_prefix = f"{detector_str}_{order_str}_T{imgtypes_str}"


    # --- Coefficient vs Segment Plots ---
    # Only generate if there's more than one segment and enabled
    if len(results_table) > 1 and comparison_plots_enabled:
         print("Generating comparison plots...")
         # Pass the table directly to plotting functions if they support it, or convert
         # For now, convert to list of dicts as plotting funcs expect that structure
         results_list_for_plotting = [{name: row[name] for name in results_table.dtype.names} for row in results_table]

         fname_defocus = comp_subdir / f"{file_prefix}_analysis_defocus.{plot_format}" if save_plots else None
         fname_astig = comp_subdir / f"{file_prefix}_analysis_astig.{plot_format}" if save_plots else None
         fname_coma = comp_subdir / f"{file_prefix}_analysis_coma.{plot_format}" if save_plots else None
         try: plot_coefficient_vs_segment(results_list_for_plotting, zernike_indices, 2, 0, title=f"Defocus {file_prefix}", filename=fname_defocus)
         except Exception as e: print(f"Error plotting Defocus: {e}")
         try: plot_coefficient_vs_segment(results_list_for_plotting, zernike_indices, 2, 2, title=f"Astigmatism {file_prefix}", filename=fname_astig)
         except Exception as e: print(f"Error plotting Astigmatism: {e}")
         try: plot_coefficient_vs_segment(results_list_for_plotting, zernike_indices, 3, 1, title=f"Coma {file_prefix}", filename=fname_coma)
         except Exception as e: print(f"Error plotting Coma: {e}")

         fname_rmse = comp_subdir / f"{file_prefix}_analysis_rmse_diff.{plot_format}" if save_plots else None
         fname_cos = comp_subdir / f"{file_prefix}_analysis_cosine_sim.{plot_format}" if save_plots else None
         try: plot_adjacent_segment_comparison(results_list_for_plotting, metric='rmse_diff', title=f"RMSE Diff {file_prefix}", filename=fname_rmse)
         except Exception as e: print(f"Error plotting RMSE Diff: {e}")
         try: plot_adjacent_segment_comparison(results_list_for_plotting, metric='cosine_similarity', title=f"Cosine Sim {file_prefix}", filename=fname_cos)
         except Exception as e: print(f"Error plotting Cosine Sim: {e}")



    # --- Heatmap Plot ---
    if len(results_table) > 1 and heatmap_plot_enabled:
        print("Generating coefficient heatmap...")
        fn_heatmap = comp_subdir / f"{file_prefix}_analysis_heatmap.{plot_format}" if save_plots else None
        results_list_for_plotting = [{name: row[name] for name in results_table.dtype.names} for row in results_table] # Convert for now
        try:
            plot_coefficient_heatmap_vs_segment(
                results_list_for_plotting, zernike_indices,
                value_type=heatmap_config.get('value_type', 'coeff'),
                title=f"Coeff Heatmap {file_prefix}", filename=fn_heatmap)
        except Exception as e: print(f"Error plotting Heatmap: {e}")

    # --- 2D Detector Map Plots ---
    if len(results_table) > 0 and detector_plots_enabled: # Need at least one segment with position
        # Check if MEDIAN_X/Y columns exist and are not all NaN
        if 'MEDIAN_X' in results_table.dtype.names and np.any(np.isfinite(results_table['MEDIAN_X'])):
             print("Generating 2D detector coefficient maps...")
             value_type_2d = detector_map_config.get('value_type', 'coeff')
             coeffs_to_plot = [(0, 0, "Piston"), 
                               (1, -1, "Horizontal tilt"), 
                               (1, 1, "Vertical tilt"), 
                               (2, 0, "Defocus"), 
                               (2, 2, "Astigmatism vertical"), 
                               (2, -2, "Astigmatism oblique"),
                               (3, 1, "Primary Coma Horizontal"), 
                               (3,-1, "Primary Coma Oblique"),
                               (3, -3, "Trefoil vertical"), 
                               (3, 3, "Trefoil oblique"), 
                               (4, 0, "Primary spherical"),
                               (4, -2, "Secondary astigmatism oblique"), 
                               (4, 2, "Secondary astigmatism vertical"),
                               (4, -4, "Quadrafoil vertical"), 
                               (4, 4, "Quadrafoil oblique"),
                               (5, -1, "Secondary coma horizontal"), 
                               (5, 1, "Secondary coma vertical"), 
                               (6, 0, "Secondary spherical"),
                               (6, 2, "Tertiary astigmatism oblique"),
                               ]
             for n, m, name in coeffs_to_plot:
                 fn_2d = detector_map_dir / f"{file_prefix}_map_{name}_Z{n}{m}.{plot_format}" if save_plots else None
                 title_2d = f"{name} Z({n},{m}) Variation ({value_type_2d}) - {file_prefix}"
                 try:
                     plot_coefficient_on_detector(
                          results_table, # Pass the table directly
                          zernike_indices,
                          coeff_n=n, coeff_m=m, value_type=value_type_2d,
                          title=title_2d, filename=fn_2d, img_shape=img_shape)
                 except Exception as e: print(f"Error plotting Detector Map for Z({n},{m}): {e}")
        else:
             print("Cannot generate 2D detector maps: MEDIAN_X/Y columns missing or all NaN.")


    print("--- Phase 2 Plotting Complete ---")


# --- Main Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Echelle 2D Zernike Analysis")
    parser.add_argument("lfc_file", type=str, help="Path to the input LFC FITS file.")
    parser.add_argument("-b", "--bias", type=str, default=None, help="Path to the bias FITS file (optional).")
    parser.add_argument("-O", "--output_dir", type=str, default="./echelle_analysis_output", help="Directory for output FITS and plots.")
    parser.add_argument("-d", "--detector", type=str, default="red", choices=["red", "blue"], help="Detector to analyze ('red' or 'blue').")
    parser.add_argument("--orders", type=int, nargs='+', default=None, help="Specific order numbers to process (e.g., --orders 4 5 6). Processes all if omitted.")
    parser.add_argument("--images", type=str, nargs='+', default=['A','B'], choices=['A', 'B'], help="Image types to process ('A', 'B').")
    parser.add_argument("--nmax", type=int, default=6, help="Maximum Zernike radial order (n_max).")
    parser.add_argument("--rmax", type=float, default=5.0, help="Zernike normalization radius (r_max).")
    parser.add_argument("--segments", type=int, default=16, help="Number of segments per order.")
    parser.add_argument("--clobber", action='store_true', help="Overwrite existing output FITS file.")
    parser.add_argument("--skip_analysis", action='store_true', help="Skip the analysis phase and only run post-plotting (requires existing FITS file).")
    parser.add_argument("--skip_plotting", action='store_true', help="Skip the post-analysis plotting phase.")
    # Plotting flags (control enabling plot types)
    parser.add_argument("--show_plots", action='store_true', help="Show plots interactively instead of saving to file.")
    parser.add_argument("--plot_format", type=str, default="pdf", choices=["pdf", "png"], help="Output format for saved plots.")
    parser.add_argument("--plot_stamps", action='store_true', help="Generate per-segment stamp/residual plots.")
    parser.add_argument("--plot_zernike", action='store_true', help="Generate per-segment Zernike fit/residual plots.")
    parser.add_argument("--plot_spectrum", action='store_true', help="Generate per-segment Zernike spectrum plots.")
    parser.add_argument("--plot_comparison", action='store_true', help="Generate cross-segment coefficient vs segment plots.")
    parser.add_argument("--plot_heatmap", action='store_true', help="Generate coefficient heatmap plot.")
    parser.add_argument("--plot_detector", action='store_true', help="Generate 2D detector map plots.")

    args = parser.parse_args()

    # --- Build Configuration Dictionary from Args ---
    config = {
        'lfc_filename': Path(args.lfc_file),
        'bias_filename': Path(args.bias) if args.bias else None,
        'output_dir': Path(args.output_dir),
        'detector': args.detector,
        'hard_flux_cut': 320 if args.detector == 'red' else 420,
        'clobber_output': args.clobber,

        'analysis_params': {
            'peak_min_dist': 5, # Default
            'cluster_eps_x': 0.002 if args.detector == 'red' else 0.004, # Default
            'cluster_eps_y': 0.02 if args.detector == 'red' else 0.008, # Default
            'cluster_min_samp': 5, # Default
            'stamp_hw': 5, # Default
            'fit_snr_thresh': 5.0, # Default
            'num_seg': args.segments,
            'n_max_z': args.nmax,
            'r_max_z': args.rmax,
            'peak_threshold_abs': None, # Default
            'peak_threshold_rel': None, # Default
        },

        'processing_control': {
            'orders_to_run': args.orders,
            'images_to_run': args.images
        },

        'plot_config': {
            'save_plots': not args.show_plots,
            'plot_format': args.plot_format.lower(),
            'plot_dir': Path(args.output_dir) / "plots",
            'interactive_peaks': False, # Default
            'interactive_clusters': False, # Default
            # Enable flags based on command line args
            'stamps': {'enable': args.plot_stamps},
            'zernike': {'enable': args.plot_zernike},
            'spectrum': {'enable': args.plot_spectrum, 'spectrum_type': 'coeff'}, # Default spectrum type
            'comparison': {'enable': args.plot_comparison},
            'heatmap': {'enable': args.plot_heatmap, 'value_type': 'coeff'}, # Default heatmap type
            'detector_map': {'enable': args.plot_detector, 'value_type': 'coeff'} # Default detector map type
        }
    }
    # Define expected output path based on config for skipping analysis phase
    output_fits_path = config['output_dir'] / f"{config['lfc_filename'].stem}_{config['detector']}_zernike_results.fits"
    config['output_fits_path'] = output_fits_path # Add path to config for easy access

    # --- Execute Phases ---
    final_fits_path = None
    if not args.skip_analysis:
        # run_analysis now also handles writing the peak catalog internally
        final_fits_path = run_analysis(config)
    else:
        # ... (Logic to set final_fits_path if skipping analysis) ...
        output_fits_path = config['output_fits_path']
        print(f"Skipping analysis phase. Assuming file exists at: {output_fits_path}")
        if output_fits_path.exists(): final_fits_path = output_fits_path
        else: print(f"Error: Cannot skip analysis, FITS file not found at {output_fits_path}")

    # --- Post-Plotting Phase ---
    if not args.skip_plotting and final_fits_path:
        # Pass the path and config
        run_post_analysis_plotting(final_fits_path, config)
    elif args.skip_plotting: print("Skipping post-analysis plotting phase.")
    else: print("Cannot run post-analysis plotting as FITS file was not found/created.")

    # --- Example Usage of New Reading Functions (after FITS exists) ---
    if final_fits_path and final_fits_path.exists():
         print("\n--- Example Reading Peak Catalog ---")
         # Create an instance to use the reading methods
         reader_analyzer = EchelleAnalyzer(config['lfc_filename'])
         reader_analyzer.output_fits_path = final_fits_path # Point to the results file

         # Read the whole catalog
         peak_catalog = reader_analyzer.read_peak_catalog()
         if peak_catalog is not None:
              print(f"Read {len(peak_catalog)} total peaks from catalog.")

              # Get peaks for a specific segment
              example_order = config['processing_control']['orders_to_run'][0] if config['processing_control']['orders_to_run'] else 1 # Get first order run/default
              example_image_str = config['processing_control']['images_to_run'][0]
              example_image_int = 0 if example_image_str == 'A' else 1
              example_segment = 5

              segment_peaks_coords = reader_analyzer.get_peaks_for_segment(
                  example_order, example_image_int, example_segment
              )
              if segment_peaks_coords is not None:
                   print(f"Retrieved {len(segment_peaks_coords)} peaks for O={example_order}, T={example_image_str}, S={example_segment}")

                   # Now you could loop through these coords, get stamp data, and fit with BayesianStampFitter
                   if len(segment_peaks_coords) > 0:
                       first_peak_x, first_peak_y = segment_peaks_coords[0]
                       print(f"  First peak coords: ({first_peak_x}, {first_peak_y})")
                       # Need to load data if not already loaded in this scope
                       # stamp_data = reader_analyzer.get_stamp_data(first_peak_x, first_peak_y)
                       # if stamp_data is not None:
                       #     print(f"  Retrieved stamp data of shape: {stamp_data.shape}")
                           # bayesian_fitter = BayesianStampFitter(...)
                           # prior_info = ... # Load/generate prior info
                           # bayesian_fitter.fit(stamp_data, first_peak_x, first_peak_y, prior_info)


    print("\nWorkflow finished.")


