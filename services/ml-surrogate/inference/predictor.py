"""
ONNX-based ML Surrogate Predictor
GPU-accelerated inference for aerodynamic predictions
"""

import numpy as np
import onnxruntime as ort
from typing import Dict, Optional, Tuple
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class AeroPredictor:
    """
    ONNX Runtime predictor for aerodynamic surrogate models
    
    Features:
    - GPU acceleration via CUDA
    - Batch processing
    - Confidence estimation
    - Performance monitoring
    """
    
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        providers: Optional[list] = None
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to ONNX model file
            use_gpu: Use CUDA if available
            providers: ONNX Runtime execution providers
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Set up execution providers
        if providers is None:
            if use_gpu:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output metadata
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        logger.info(f"Predictor initialized: {model_path}")
        logger.info(f"Providers: {self.session.get_providers()}")
        logger.info(f"Inputs: {self.input_names}")
        logger.info(f"Outputs: {self.output_names}")
    
    def predict(
        self,
        mesh: np.ndarray,
        parameters: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict aerodynamic quantities
        
        Args:
            mesh: Mesh coordinates [nodes, 3] or [batch, nodes, 3]
            parameters: Flow parameters [3] or [batch, 3]
            return_confidence: Return confidence scores
            
        Returns:
            Dictionary with predictions
        """
        start_time = time.time()
        
        # Ensure batch dimension
        if mesh.ndim == 2:
            mesh = mesh[np.newaxis, ...]
        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, ...]
        
        # Prepare inputs
        inputs = {
            self.input_names[0]: mesh.astype(np.float32),
            self.input_names[1]: parameters.astype(np.float32)
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Parse outputs
        result = {}
        for name, value in zip(self.output_names, outputs):
            result[name] = value
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Add metadata
        result['inference_time_ms'] = inference_time * 1000
        result['batch_size'] = mesh.shape[0]
        
        logger.debug(f"Inference completed in {inference_time*1000:.2f}ms")
        
        return result
    
    def predict_batch(
        self,
        meshes: list,
        parameters_list: list,
        batch_size: int = 32
    ) -> list:
        """
        Batch prediction for multiple designs
        
        Args:
            meshes: List of mesh arrays
            parameters_list: List of parameter arrays
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        n_samples = len(meshes)
        
        for i in range(0, n_samples, batch_size):
            batch_meshes = meshes[i:i+batch_size]
            batch_params = parameters_list[i:i+batch_size]
            
            # Stack into batch
            mesh_batch = np.stack(batch_meshes)
            param_batch = np.stack(batch_params)
            
            # Predict
            batch_result = self.predict(mesh_batch, param_batch)
            
            # Split batch results
            for j in range(len(batch_meshes)):
                sample_result = {
                    key: value[j] if value.ndim > 0 else value
                    for key, value in batch_result.items()
                }
                results.append(sample_result)
        
        logger.info(f"Batch prediction completed: {n_samples} samples")
        
        return results
    
    def estimate_confidence(
        self,
        mesh: np.ndarray,
        parameters: np.ndarray
    ) -> float:
        """
        Estimate prediction confidence
        
        Methods:
        - Model uncertainty (if available)
        - Distance to training data
        - Output consistency checks
        
        Args:
            mesh: Mesh coordinates
            parameters: Flow parameters
            
        Returns:
            Confidence score [0, 1]
        """
        result = self.predict(mesh, parameters)
        
        # If model outputs confidence directly
        if 'confidence' in result:
            return float(result['confidence'])
        
        # Otherwise, use heuristics
        confidence = 1.0
        
        # Check if outputs are reasonable
        if 'cl' in result:
            cl = float(result['cl'])
            if cl < -1.0 or cl > 3.0:  # Unrealistic CL
                confidence *= 0.5
        
        if 'cd' in result:
            cd = float(result['cd'])
            if cd < 0.0 or cd > 1.0:  # Unrealistic CD
                confidence *= 0.5
        
        if 'cl' in result and 'cd' in result:
            if result['cd'] > result['cl']:  # Drag > Lift (unusual)
                confidence *= 0.7
        
        return confidence
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        if self.inference_count == 0:
            return {
                'total_inferences': 0,
                'avg_inference_time_ms': 0.0,
                'total_time_s': 0.0
            }
        
        return {
            'total_inferences': self.inference_count,
            'avg_inference_time_ms': (self.total_inference_time / self.inference_count) * 1000,
            'total_time_s': self.total_inference_time
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0
        logger.info("Performance statistics reset")


class PredictorCache:
    """
    Cache for prediction results
    Reduces redundant computations
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_key(self, mesh: np.ndarray, parameters: np.ndarray) -> str:
        """Generate cache key from inputs"""
        mesh_hash = hash(mesh.tobytes())
        param_hash = hash(parameters.tobytes())
        return f"{mesh_hash}_{param_hash}"
    
    def get(self, mesh: np.ndarray, parameters: np.ndarray) -> Optional[Dict]:
        """Get cached result"""
        key = self.get_key(mesh, parameters)
        
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, mesh: np.ndarray, parameters: np.ndarray, result: Dict):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_key(mesh, parameters)
        self.cache[key] = result
        logger.debug(f"Cached result: {key}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")


if __name__ == "__main__":
    # Test predictor
    logging.basicConfig(level=logging.INFO)
    
    # Note: This requires an actual ONNX model file
    # For testing, we'll just show the interface
    
    print("AeroPredictor Test Interface")
    print("=" * 50)
    print("\nUsage:")
    print("  predictor = AeroPredictor('model.onnx', use_gpu=True)")
    print("  result = predictor.predict(mesh, parameters)")
    print("\nFeatures:")
    print("  - GPU acceleration (CUDA)")
    print("  - Batch processing")
    print("  - Confidence estimation")
    print("  - Performance tracking")
    print("  - Result caching")
