#!/usr/bin/env python3
"""
Machine learning surrogate models for analog Hawking radiation detection.
Implements Gaussian Process and Neural Network surrogates for fast parameter
exploration and sensitivity analysis.
"""

import numpy as np
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import argparse
from dataclasses import dataclass

# Machine learning libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@dataclass
class SurrogateConfig:
    """Configuration for surrogate models"""
    model_type: str = "gp"  # "gp", "rf", "nn", "ensemble"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100  # For RF
    max_depth: int = 10  # For RF
    hidden_layers: List[int] = None  # For NN
    learning_rate: float = 0.001  # For NN
    epochs: int = 1000  # For NN
    batch_size: int = 32  # For NN
    kernel_type: str = "matern"  # For GP: "rbf", "matern"
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 128, 64]

class HawkingDataset(Dataset):
    """PyTorch dataset for Hawking radiation data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HawkingNeuralNetwork(nn.Module):
    """Neural network surrogate model"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int = 1):
        super(HawkingNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SurrogateTrainer:
    """Trains and evaluates surrogate models for Hawking radiation detection"""
    
    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("results/surrogates")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'surrogate_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load training data from sweep results"""
        
        data_path = Path(data_path)
        
        if data_path.is_file():
            files = [data_path]
        else:
            files = list(data_path.glob("*.json"))
        
        X_list = []
        y_list = []
        
        feature_names = [
            'laser_intensity', 'plasma_density', 'magnetic_field',
            'temperature_constant', 'laser_wavelength', 'grid_max',
            'mirror_D', 'mirror_eta'
        ]
        
        for file in files:
            try:
                with open(file) as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                
                for result in results:
                    if 'error' in result:
                        continue
                    
                    params = result.get('parameters', {})
                    
                    # Extract features
                    features = []
                    for name in feature_names:
                        if name in params:
                            features.append(float(params[name]))
                        elif name == 'magnetic_field' and 'magnetic_field' not in params:
                            features.append(0.0)  # Default for None
                        elif name in ['mirror_D', 'mirror_eta'] and name not in params:
                            features.append(10e-6 if name == 'mirror_D' else 1.0)  # Defaults
                        else:
                            features.append(0.0)
                    
                    # Extract target (detection time)
                    target = result.get('t5sigma_s', None)
                    if target is not None and np.isfinite(target):
                        X_list.append(features)
                        y_list.append(float(target))
            
            except Exception as e:
                self.logger.warning(f"Error loading {file}: {e}")
                continue
        
        if not X_list:
            raise ValueError("No valid training data found")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Log-transform detection times for better scaling
        y = np.log10(np.maximum(y, 1e-10))
        
        self.logger.info(f"Loaded {len(X)} training samples")
        self.logger.info(f"Feature names: {feature_names}")
        
        return X, y, feature_names
    
    def train_gaussian_process(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Gaussian Process surrogate model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define kernel
        if self.config.kernel_type == "rbf":
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * X.shape[1], (1e-2, 1e2)) + WhiteKernel(noise_level=1.0)
        else:  # matern
            kernel = C(1.0, (1e-3, 1e3)) * Matern([1.0] * X.shape[1], (1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1.0)
        
        # Train model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=self.config.random_state
        )
        
        gp.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = gp.predict(X_test_scaled)
        
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'cv_score': float(np.mean(cross_val_score(gp, X_train_scaled, y_train, cv=5)))
        }
        
        self.models['gp'] = gp
        self.scalers['gp'] = scaler
        self.metrics['gp'] = metrics
        
        self.logger.info(f"GP Model - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest surrogate model"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test_scaled)
        
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'cv_score': float(np.mean(cross_val_score(rf, X_train_scaled, y_train, cv=5)))
        }
        
        self.models['rf'] = rf
        self.scalers['rf'] = scaler
        self.metrics['rf'] = metrics
        
        self.logger.info(f"RF Model - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Neural Network surrogate model"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Scale features and targets
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # Convert to PyTorch datasets
        train_dataset = HawkingDataset(X_train_scaled, y_train_scaled)
        test_dataset = HawkingDataset(X_test_scaled, y_test_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HawkingNeuralNetwork(
            input_dim=X.shape[1],
            hidden_layers=self.config.hidden_layers
        ).to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        # Evaluation
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        # Inverse transform predictions
        predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
        actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).ravel()
        
        metrics = {
            'mse': float(mean_squared_error(actuals, predictions)),
            'r2': float(r2_score(actuals, predictions)),
            'final_loss': float(train_losses[-1])
        }
        
        self.models['nn'] = model
        self.scalers['nn_features'] = feature_scaler
        self.scalers['nn_target'] = target_scaler
        self.metrics['nn'] = metrics
        
        self.logger.info(f"NN Model - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ensemble of surrogate models"""
        
        # Train individual models
        metrics = {}
        
        # Gaussian Process
        gp_metrics = self.train_gaussian_process(X, y)
        metrics['gp'] = gp_metrics
        
        # Random Forest
        rf_metrics = self.train_random_forest(X, y)
        metrics['rf'] = rf_metrics
        
        # Neural Network
        nn_metrics = self.train_neural_network(X, y)
        metrics['nn'] = nn_metrics
        
        # Select best model
        best_model = min(metrics.keys(), key=lambda k: metrics[k]['mse'])
        
        metrics['best_model'] = best_model
        metrics['ensemble_metrics'] = {
            'model_comparison': metrics,
            'selected_model': best_model
        }
        
        self.logger.info(f"Best model: {best_model}")
        
        return metrics
    
    def predict(self, X: np.ndarray, model_type: str = None) -> np.ndarray:
        """Make predictions using trained surrogate model"""
        
        if model_type is None:
            model_type = self.config.model_type
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        if model_type == 'nn':
            # Neural network prediction
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.models['nn'].eval()
            
            X_scaled = self.scalers['nn_features'].transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            
            with torch.no_grad():
                predictions = self.models['nn'](X_tensor).squeeze().cpu().numpy()
            
            # Inverse transform
            predictions = self.scalers['nn_target'].inverse_transform(
                predictions.reshape(-1, 1)
            ).ravel()
            
            # Convert back from log scale
            predictions = 10**predictions
            
        else:
            # Scikit-learn models
            X_scaled = self.scalers[model_type].transform(X)
            predictions = self.models[model_type].predict(X_scaled)
            
            # Convert back from log scale
            predictions = 10**predictions
        
        return predictions
    
    def sensitivity_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Perform sensitivity analysis using surrogate models"""
        
        if 'rf' not in self.models:
            self.train_random_forest(X, np.log10(1e-10 + np.ones(len(X))))  # Dummy training
        
        rf_model = self.models['rf']
        
        # Feature importance from Random Forest
        importances = rf_model.feature_importances_
        
        # Sobol sensitivity analysis (simplified)
        # For proper Sobol analysis, use SALib
        
        sensitivity = {
            'feature_importance': dict(zip(feature_names, importances.tolist())),
            'ranked_features': sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        return sensitivity
    
    def save_models(self, output_dir: str = "results/surrogates"):
        """Save trained models and scalers"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scikit-learn models
        for model_name, model in self.models.items():
            if model_name != 'nn':  # Skip neural network for now
                model_path = output_path / f"{model_name}_model.pkl"
                joblib.dump(model, model_path)
                
                if model_name in self.scalers:
                    scaler_path = output_path / f"{model_name}_scaler.pkl"
                    joblib.dump(self.scalers[model_name], scaler_path)
        
        # Save configuration
        config_path = output_path / "surrogate_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.config.model_type,
                'metrics': self.metrics,
                'config': {
                    'test_size': self.config.test_size,
                    'random_state': self.config.random_state
                }
            }, f, indent=2)
        
        self.logger.info(f"Models saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train surrogate models for Hawking radiation detection")
    parser.add_argument("--data", required=True, help="Path to training data (JSON files)")
    parser.add_argument("--model", default="ensemble", choices=["gp", "rf", "nn", "ensemble"], 
                       help="Model type to train")
    parser.add_argument("--output", default="results/surrogates", help="Output directory")
    parser.add_argument("--predict", help="Make predictions on new parameters")
    parser.add_argument("--sensitivity", action="store_true", help="Perform sensitivity analysis")
    
    args = parser.parse_args()
    
    config = SurrogateConfig(model_type=args.model)
    trainer = SurrogateTrainer(config)
    
    # Load and prepare data
    X, y, feature_names = trainer.load_data(args.data)
    
    # Train models
    if args.model == "ensemble":
        metrics = trainer.train_ensemble(X, y)
    elif args.model == "gp":
        metrics = trainer.train_gaussian_process(X, y)
    elif args.model == "rf":
        metrics = trainer.train_random_forest(X, y)
    elif args.model == "nn":
        metrics = trainer.train_neural_network(X, y)
    
    # Save models
    trainer.save_models(args.output)
    
    # Sensitivity analysis
    if args.sensitivity:
        sensitivity = trainer.sensitivity_analysis(X, feature_names)
        
        sensitivity_path = Path(args.output) / "sensitivity_analysis.json"
        with open(sensitivity_path, 'w') as f:
            json.dump(sensitivity, f, indent=2)
        
        print("\nFeature Importance:")
        for feature, importance in sensitivity['ranked_features']:
            print(f"  {feature}: {importance:.4f}")
    
    # Make predictions if requested
    if args.predict:
        # Example usage
        sample_params = np.array([[5e17, 5e17, 0, 1e4, 800e-9, 50e-6, 10e-6, 1.0]])
        predictions = trainer.predict(sample_params)
        print(f"\nPredicted detection time: {predictions[0]:.2e} seconds")
    
    print(f"\nTraining complete. Metrics: {metrics}")

if __name__ == "__main__":
    main()
