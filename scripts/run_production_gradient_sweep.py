#!/usr/bin/env python3
"""
Production-scale gradient catastrophe sweep for publication results.

This script runs the gradient catastrophe analysis with publication-quality 
parameters (500+ configurations) to generate robust statistical results.
"""

import sys
from pathlib import Path
import subprocess
import time

def main():
    """Run production gradient catastrophe sweep"""
    
    print("🚀 PRODUCTION GRADIENT CATASTROPHE SWEEP")
    print("="*50)
    print("Running comprehensive parameter space exploration...")
    print("This may take 5-10 minutes for robust statistics.")
    
    start_time = time.time()
    
    # Run the main sweep with production parameters
    cmd = [
        sys.executable, 
        "scripts/sweep_gradient_catastrophe.py",
        "--n-samples", "500",  # Robust statistics
        "--output", "results/gradient_limits_production"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SWEEP COMPLETED SUCCESSFULLY!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Sweep failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ PRODUCTION SWEEP COMPLETE!")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Results: results/gradient_limits_production/")
    print(f"Analysis: results/gradient_limits_production/gradient_catastrophe_findings.md")
    
    # Quick summary of findings
    try:
        import json
        with open("results/gradient_limits_production/gradient_catastrophe_sweep.json") as f:
            data = json.load(f)
        
        analysis = data['analysis']
        print(f"\n🎯 PRODUCTION RESULTS SUMMARY:")
        print(f"  Maximum κ: {analysis['max_kappa']:.2e} Hz")
        print(f"  Valid configurations: {analysis['valid_configurations']}")
        print(f"  Breakdown rate: {analysis['breakdown_statistics']['total_breakdown_rate']:.1%}")
        print(f"  Optimal a₀: {analysis['max_kappa_config']['a0']:.2f}")
        print(f"  Optimal n_e: {analysis['max_kappa_config']['n_e']:.2e} m⁻³")
        
    except Exception as e:
        print(f"Could not load results summary: {e}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())