import json
import os
import glob

def find_best_parameters(results_dir):
    """
    Parses all parameter_optimization_*.json files in a directory to find the
    configuration with the highest detection confidence.

    Args:
        results_dir (str): The path to the directory containing the JSON files.

    Returns:
        dict: The best performing parameter configuration, or None if no
              successful runs are found.
    """
    best_confidence = -1
    best_run = None

    json_files = glob.glob(os.path.join(results_dir, 'parameter_optimization_*.json'))

    if not json_files:
        print("No parameter optimization files found.")
        return None

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

                # Check global optimization results first
                global_opt = data.get('global_optimization', {})
                if global_opt:
                    best_result = global_opt.get('best_result', {})
                    if best_result and best_result.get('success'):
                        best_fit = best_result.get('best_fit', {})
                        if best_fit:
                            confidence = best_fit.get('confidence', 0)
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_run = best_result

                # Then check sweep results
                sweep_results = data.get('sweep_results', {})
                for sweep_type, sweep_runs in sweep_results.items():
                    for run in sweep_runs:
                        if run.get('success'):
                            best_fit = run.get('best_fit', {})
                            if best_fit:
                                confidence = best_fit.get('confidence', 0)
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_run = run

        except (json.JSONDecodeError, IOError) as e:
            print(f"Could not read or parse {file_path}: {e}")
            continue

    return best_run

if __name__ == '__main__':
    results_directory = 'results'
    best_configuration = find_best_parameters(results_directory)

    if best_configuration:
        print("=== BEST PERFORMING PARAMETERS FOUND ===")
        params = best_configuration['parameters']
        best_fit = best_configuration['best_fit']
        derived = best_configuration['derived_metrics']

        print(f"Confidence: {best_fit.get('confidence', 0):.3f}")
        print(f"Best fit method: {best_fit.get('method', 'N/A')}")
        print(f"Temperature: {best_fit.get('temperature', 0):.2e} K")
        print(f"Reduced χ²: {best_fit.get('chi_squared', 'N/A'):.3f}")
        print("-" * 40)
        print("Parameters:")
        print(f"  Wavelength: {params.get('wavelength', 0) * 1e9:.1f} nm")
        print(f"  Pulse duration: {params.get('pulse_duration', 0) * 1e15:.1f} fs")
        print(f"  Focus diameter: {params.get('focus_diameter', 0) * 1e6:.1f} µm")
        print(f"  Pressure: {params.get('pressure', 0):.2e} Torr")
        print(f"  Gas Type: {params.get('gas_type', 'N/A')}")
        print(f"  Electric field: {params.get('e_field', 0):.2e} V/m")
        print("-" * 40)
        print("Derived Metrics:")
        print(f"  Horizon strength: {derived.get('horizon_strength', 'N/A'):.2f}")
        print(f"  Pair production rate: {derived.get('pair_production_rate', 'N/A'):.2e}")
    else:
        print("No successful optimization runs found to determine the best parameters.")
