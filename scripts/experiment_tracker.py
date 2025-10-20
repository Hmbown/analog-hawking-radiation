#!/usr/bin/env python3
"""
Experiment UUID Tracking and Manifest System

Provides comprehensive experiment tracking with UUID generation,
provenance tracking, git integration, and experiment manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentProvenance:
    """Tracks experiment provenance including git state and dependencies"""
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    python_version: str = ""
    hostname: str = ""
    username: str = ""
    start_timestamp: float = 0.0
    dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    """Tracks experiment performance and resource metrics"""
    total_simulations: int = 0
    successful_simulations: int = 0
    failed_simulations: int = 0
    total_compute_time: float = 0.0
    average_simulation_time: float = 0.0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    disk_usage_gb: float = 0.0


@dataclass
class ExperimentManifest:
    """Comprehensive experiment manifest with tracking information"""
    experiment_id: str
    name: str
    description: str
    version: str = "1.0.0"
    status: str = "created"  # created, running, completed, failed, cancelled
    
    # Timestamps
    created_timestamp: float = 0.0
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    last_updated: float = 0.0
    
    # Experiment structure
    phases: List[str] = field(default_factory=list)
    current_phase: Optional[str] = None
    phase_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tracking information
    provenance: ExperimentProvenance = field(default_factory=ExperimentProvenance)
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    configuration_files: List[str] = field(default_factory=list)
    result_files: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    priority: str = "normal"  # low, normal, high, critical


class ExperimentTracker:
    """Manages experiment tracking, UUID generation, and manifest management"""
    
    def __init__(self, experiment_name: Optional[str] = None, base_dir: str = "results/orchestration"):
        self.base_dir = Path(base_dir)
        self.experiment_id = self._generate_experiment_id()
        
        if experiment_name is None:
            experiment_name = f"hawking_experiment_{self.experiment_id}"
        
        self.manifest = ExperimentManifest(
            experiment_id=self.experiment_id,
            name=experiment_name,
            description="Analog Hawking Radiation Detection Experiment",
            created_timestamp=time.time(),
            last_updated=time.time()
        )
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize provenance
        self._capture_provenance()
        
        # Create experiment directory
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized experiment tracker for {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID with timestamp"""
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"exp_{timestamp}_{unique_id}"
    
    def _setup_logging(self) -> None:
        """Setup experiment-specific logging"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(experiment_id)s] %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'experiment_{self.experiment_id}.log'),
                logging.StreamHandler()
            ]
        )
        
        # Add experiment ID to log records
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.experiment_id = self.experiment_id
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    def _capture_provenance(self) -> None:
        """Capture system and git provenance information"""
        self.manifest.provenance.start_timestamp = time.time()
        self.manifest.provenance.python_version = sys.version
        self.manifest.provenance.hostname = os.uname().nodename if hasattr(os, 'uname') else "unknown"
        self.manifest.provenance.username = os.getenv('USER', 'unknown')
        
        # Capture git information
        try:
            # Git commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                self.manifest.provenance.git_commit = result.stdout.strip()
            
            # Git branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                self.manifest.provenance.git_branch = result.stdout.strip()
            
            # Git dirty status
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )
            self.manifest.provenance.git_dirty = bool(result.stdout.strip())
            
        except Exception as e:
            self.logger.warning(f"Could not capture git information: {e}")
        
        # Capture dependency versions
        try:
            import pkg_resources
            dependencies = ['numpy', 'scipy', 'matplotlib', 'scikit-optimize']
            for dep in dependencies:
                try:
                    version = pkg_resources.get_distribution(dep).version
                    self.manifest.provenance.dependencies[dep] = version
                except:
                    self.manifest.provenance.dependencies[dep] = "not_found"
        except:
            self.logger.warning("Could not capture dependency versions")
    
    def start_experiment(self) -> None:
        """Mark experiment as started"""
        self.manifest.status = "running"
        self.manifest.start_timestamp = time.time()
        self.manifest.last_updated = time.time()
        
        self.logger.info(f"Started experiment: {self.manifest.name}")
        self._save_manifest()
    
    def complete_experiment(self) -> None:
        """Mark experiment as completed"""
        self.manifest.status = "completed"
        self.manifest.end_timestamp = time.time()
        self.manifest.last_updated = time.time()
        
        # Calculate final metrics
        if self.manifest.metrics.total_simulations > 0:
            self.manifest.metrics.average_simulation_time = (
                self.manifest.metrics.total_compute_time / 
                self.manifest.metrics.total_simulations
            )
        
        self.logger.info(f"Completed experiment: {self.manifest.name}")
        self._save_manifest()
    
    def fail_experiment(self, error_message: str) -> None:
        """Mark experiment as failed"""
        self.manifest.status = "failed"
        self.manifest.end_timestamp = time.time()
        self.manifest.last_updated = time.time()
        self.manifest.notes = f"Experiment failed: {error_message}"
        
        self.logger.error(f"Experiment failed: {error_message}")
        self._save_manifest()
    
    def add_phase(self, phase_name: str, phase_config: Dict[str, Any]) -> None:
        """Add a phase to the experiment"""
        if phase_name not in self.manifest.phases:
            self.manifest.phases.append(phase_name)
        
        phase_record = {
            "phase_name": phase_name,
            "start_timestamp": time.time(),
            "config": phase_config,
            "status": "pending"
        }
        
        self.manifest.phase_history.append(phase_record)
        self.manifest.last_updated = time.time()
        
        self.logger.info(f"Added phase: {phase_name}")
        self._save_manifest()
    
    def start_phase(self, phase_name: str) -> None:
        """Mark a phase as started"""
        self.manifest.current_phase = phase_name
        
        for phase_record in self.manifest.phase_history:
            if phase_record["phase_name"] == phase_name:
                phase_record["status"] = "running"
                phase_record["start_timestamp"] = time.time()
                break
        
        self.manifest.last_updated = time.time()
        
        self.logger.info(f"Started phase: {phase_name}")
        self._save_manifest()
    
    def complete_phase(self, phase_name: str, phase_results: Dict[str, Any]) -> None:
        """Mark a phase as completed"""
        for phase_record in self.manifest.phase_history:
            if phase_record["phase_name"] == phase_name:
                phase_record["status"] = "completed"
                phase_record["end_timestamp"] = time.time()
                phase_record["results_summary"] = {
                    "total_simulations": len(phase_results.get("results", [])),
                    "successful_simulations": sum(1 for r in phase_results.get("results", []) 
                                                if r.get("simulation_success")),
                    "best_detection_time": phase_results.get("best_detection_time"),
                    "best_kappa": phase_results.get("best_kappa")
                }
                break
        
        self.manifest.last_updated = time.time()
        
        self.logger.info(f"Completed phase: {phase_name}")
        self._save_manifest()
    
    def update_metrics(self, 
                      simulations_completed: int = 0,
                      simulations_successful: int = 0,
                      compute_time: float = 0.0,
                      memory_usage_mb: Optional[float] = None,
                      cpu_usage_percent: Optional[float] = None) -> None:
        """Update experiment metrics"""
        self.manifest.metrics.total_simulations += simulations_completed
        self.manifest.metrics.successful_simulations += simulations_successful
        self.manifest.metrics.failed_simulations += (simulations_completed - simulations_successful)
        self.manifest.metrics.total_compute_time += compute_time
        
        if memory_usage_mb is not None:
            self.manifest.metrics.memory_usage_mb.append(memory_usage_mb)
        
        if cpu_usage_percent is not None:
            self.manifest.metrics.cpu_usage_percent.append(cpu_usage_percent)
        
        self.manifest.last_updated = time.time()
        
        # Auto-save every 10 metric updates or if significant changes
        if self.manifest.metrics.total_simulations % 10 == 0:
            self._save_manifest()
    
    def add_configuration_file(self, config_path: str) -> None:
        """Add a configuration file to the manifest"""
        abs_path = str(Path(config_path).resolve())
        if abs_path not in self.manifest.configuration_files:
            self.manifest.configuration_files.append(abs_path)
            self.manifest.last_updated = time.time()
    
    def add_result_file(self, result_path: str) -> None:
        """Add a result file to the manifest"""
        abs_path = str(Path(result_path).resolve())
        if abs_path not in self.manifest.result_files:
            self.manifest.result_files.append(abs_path)
            self.manifest.last_updated = time.time()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment"""
        if tag not in self.manifest.tags:
            self.manifest.tags.append(tag)
            self.manifest.last_updated = time.time()
    
    def add_note(self, note: str) -> None:
        """Add a note to the experiment"""
        if self.manifest.notes:
            self.manifest.notes += f"\n{datetime.now().isoformat()}: {note}"
        else:
            self.manifest.notes = f"{datetime.now().isoformat()}: {note}"
        
        self.manifest.last_updated = time.time()
    
    def _save_manifest(self) -> None:
        """Save the manifest to disk"""
        manifest_path = self.experiment_dir / "experiment_manifest.json"
        
        # Convert to serializable format
        manifest_dict = asdict(self.manifest)
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2, default=str)
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get the current manifest as a dictionary"""
        return asdict(self.manifest)
    
    def generate_experiment_hash(self) -> str:
        """Generate a hash representing the experiment configuration"""
        config_data = {
            "experiment_id": self.manifest.experiment_id,
            "phases": self.manifest.phases,
            "provenance": asdict(self.manifest.provenance),
            "configuration_files": self.manifest.configuration_files
        }
        
        # Include content of configuration files in hash
        for config_file in self.manifest.configuration_files:
            try:
                with open(config_file, 'r') as f:
                    config_data[config_file] = f.read()
            except:
                config_data[config_file] = "unreadable"
        
        config_json = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]
    
    def generate_provenance_report(self) -> str:
        """Generate a human-readable provenance report"""
        provenance = self.manifest.provenance
        
        report = f"Experiment Provenance Report\n"
        report += "=" * 40 + "\n\n"
        report += f"Experiment ID: {self.manifest.experiment_id}\n"
        report += f"Name: {self.manifest.name}\n"
        report += f"Status: {self.manifest.status}\n\n"
        
        report += "Git Information:\n"
        report += f"  Commit: {provenance.git_commit or 'Unknown'}\n"
        report += f"  Branch: {provenance.git_branch or 'Unknown'}\n"
        report += f"  Dirty: {provenance.git_dirty}\n\n"
        
        report += "System Information:\n"
        report += f"  Hostname: {provenance.hostname}\n"
        report += f"  Username: {provenance.username}\n"
        report += f"  Python: {provenance.python_version}\n\n"
        
        report += "Dependencies:\n"
        for dep, version in provenance.dependencies.items():
            report += f"  {dep}: {version}\n"
        
        report += f"\nStart Time: {datetime.fromtimestamp(provenance.start_timestamp).isoformat()}\n"
        
        return report
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment"""
        duration = None
        if self.manifest.start_timestamp and self.manifest.end_timestamp:
            duration = self.manifest.end_timestamp - self.manifest.start_timestamp
        elif self.manifest.start_timestamp:
            duration = time.time() - self.manifest.start_timestamp
        
        success_rate = 0.0
        if self.manifest.metrics.total_simulations > 0:
            success_rate = (self.manifest.metrics.successful_simulations / 
                          self.manifest.metrics.total_simulations)
        
        return {
            "experiment_id": self.manifest.experiment_id,
            "name": self.manifest.name,
            "status": self.manifest.status,
            "duration_seconds": duration,
            "phases_completed": len([p for p in self.manifest.phase_history 
                                   if p.get("status") == "completed"]),
            "total_phases": len(self.manifest.phases),
            "current_phase": self.manifest.current_phase,
            "total_simulations": self.manifest.metrics.total_simulations,
            "success_rate": success_rate,
            "total_compute_time": self.manifest.metrics.total_compute_time,
            "tags": self.manifest.tags
        }


class ExperimentRegistry:
    """Manages a registry of all experiments"""
    
    def __init__(self, registry_path: str = "results/orchestration/experiment_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the experiment registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_registry(self) -> None:
        """Save the registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_experiment(self, tracker: ExperimentTracker) -> None:
        """Register an experiment in the registry"""
        summary = tracker.get_experiment_summary()
        self.registry[tracker.experiment_id] = summary
        self._save_registry()
    
    def update_experiment(self, tracker: ExperimentTracker) -> None:
        """Update an experiment in the registry"""
        summary = tracker.get_experiment_summary()
        self.registry[tracker.experiment_id] = summary
        self._save_registry()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get an experiment from the registry"""
        return self.registry.get(experiment_id)
    
    def list_experiments(self, status: Optional[str] = None, 
                        tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        experiments = list(self.registry.values())
        
        if status:
            experiments = [e for e in experiments if e.get("status") == status]
        
        if tags:
            experiments = [e for e in experiments 
                          if any(tag in e.get("tags", []) for tag in tags)]
        
        return sorted(experiments, key=lambda x: x.get("experiment_id", ""))
    
    def search_experiments(self, query: str) -> List[Dict[str, Any]]:
        """Search experiments by name or ID"""
        query = query.lower()
        results = []
        
        for experiment in self.registry.values():
            if (query in experiment.get("name", "").lower() or 
                query in experiment.get("experiment_id", "").lower()):
                results.append(experiment)
        
        return results


def main():
    """Demo and testing for experiment tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Tracking System")
    parser.add_argument("--create", help="Create a new experiment with given name")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--status", help="Get status of specific experiment")
    parser.add_argument("--provenance", help="Generate provenance report for experiment")
    
    args = parser.parse_args()
    
    if args.create:
        tracker = ExperimentTracker(experiment_name=args.create)
        tracker.start_experiment()
        
        # Register in global registry
        registry = ExperimentRegistry()
        registry.register_experiment(tracker)
        
        print(f"Created experiment: {tracker.experiment_id}")
        print(f"Name: {tracker.manifest.name}")
        print(f"Directory: {tracker.experiment_dir}")
    
    elif args.list:
        registry = ExperimentRegistry()
        experiments = registry.list_experiments()
        
        print("Registered Experiments:")
        print("-" * 80)
        for exp in experiments:
            print(f"ID: {exp['experiment_id']}")
            print(f"Name: {exp['name']}")
            print(f"Status: {exp['status']}")
            print(f"Simulations: {exp['total_simulations']} (Success: {exp['success_rate']:.1%})")
            print(f"Phases: {exp['phases_completed']}/{exp['total_phases']}")
            print()
    
    elif args.status:
        registry = ExperimentRegistry()
        experiment = registry.get_experiment(args.status)
        
        if experiment:
            print("Experiment Status:")
            for key, value in experiment.items():
                print(f"  {key}: {value}")
        else:
            print(f"Experiment {args.status} not found")
    
    elif args.provenance:
        # This would require loading the full manifest
        print("Provenance report generation requires manifest file access")
        print("Use the experiment tracker within the orchestration system for full functionality")
    
    else:
        print("Experiment Tracking System")
        print("Use --create to create a new experiment")
        print("Use --list to list all experiments")
        print("Use --status <id> to get experiment status")


if __name__ == "__main__":
    main()
