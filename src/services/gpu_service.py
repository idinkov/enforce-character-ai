"""
GPU Management Service
Handles GPU detection, selection, and management across the application.
"""
import threading
from typing import List, Dict, Optional, Any, Callable

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    pynvml = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class GPUInfo:
    """Information about a single GPU device."""

    def __init__(self, index: int, name: str, memory_total: int = 0,
                 compute_capability: tuple = (0, 0)):
        self.index = index
        self.name = name
        self.memory_total = memory_total  # in bytes
        self.compute_capability = compute_capability
        self.memory_used = 0
        self.memory_free = 0
        self.utilization = 0

    def update_stats(self):
        """Update current GPU statistics."""
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.memory_used = meminfo.used
                self.memory_free = meminfo.free
                self.memory_total = meminfo.total

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.utilization = util.gpu
            except Exception:
                pass

    def get_memory_usage_percent(self) -> float:
        """Get memory usage as percentage."""
        if self.memory_total > 0:
            return (self.memory_used / self.memory_total) * 100
        return 0.0

    def get_memory_free_gb(self) -> float:
        """Get free memory in GB."""
        return self.memory_free / (1024**3)

    def get_memory_total_gb(self) -> float:
        """Get total memory in GB."""
        return self.memory_total / (1024**3)


class GPUService:
    """Service for managing GPU devices and selection across the application."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.gpus: List[GPUInfo] = []
        self.selected_gpu_index: int = -1  # -1 means CPU, 0+ means GPU index
        self.callbacks: List[callable] = []
        self._lock = threading.Lock()

        # Detect available GPUs
        self._detect_gpus()

        # Set default selection (first GPU if available, otherwise CPU)
        if self.gpus:
            self.selected_gpu_index = 0
        else:
            self.selected_gpu_index = -1

    def _detect_gpus(self):
        """Detect all available GPU devices."""
        self.gpus.clear()

        # Detect CUDA GPUs via PyTorch
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total=props.total_memory,
                        compute_capability=(props.major, props.minor)
                    )
                    gpu_info.update_stats()
                    self.gpus.append(gpu_info)
                except Exception as e:
                    print(f"Error getting info for GPU {i}: {e}")

        # If PyTorch didn't detect GPUs but NVML is available, try NVML
        elif NVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        gpu_info = GPUInfo(
                            index=i,
                            name=name,
                            memory_total=meminfo.total
                        )
                        gpu_info.update_stats()
                        self.gpus.append(gpu_info)
                    except Exception as e:
                        print(f"Error getting info for GPU {i}: {e}")
            except Exception as e:
                print(f"Error detecting GPUs with NVML: {e}")

    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get list of available devices (CPU + GPUs)."""
        devices = [{"index": -1, "name": "CPU", "type": "cpu"}]

        for gpu in self.gpus:
            devices.append({
                "index": gpu.index,
                "name": f"GPU {gpu.index}: {gpu.name}",
                "type": "gpu",
                "memory_gb": gpu.get_memory_total_gb(),
                "free_gb": gpu.get_memory_free_gb()
            })

        return devices

    def select_device(self, device_index: int) -> bool:
        """Select a device for processing (-1 for CPU, 0+ for GPU)."""
        with self._lock:
            if device_index == -1:  # CPU
                self.selected_gpu_index = -1
                self._notify_callbacks()
                return True
            elif 0 <= device_index < len(self.gpus):
                self.selected_gpu_index = device_index
                self._notify_callbacks()
                return True
            else:
                return False

    def get_selected_device(self) -> Dict[str, Any]:
        """Get currently selected device information."""
        if self.selected_gpu_index == -1:
            return {"index": -1, "name": "CPU", "type": "cpu"}
        elif 0 <= self.selected_gpu_index < len(self.gpus):
            gpu = self.gpus[self.selected_gpu_index]
            return {
                "index": gpu.index,
                "name": f"GPU {gpu.index}: {gpu.name}",
                "type": "gpu",
                "memory_gb": gpu.get_memory_total_gb(),
                "free_gb": gpu.get_memory_free_gb()
            }
        else:
            # Fallback to CPU if selection is invalid
            self.selected_gpu_index = -1
            return {"index": -1, "name": "CPU", "type": "cpu"}

    def get_torch_device(self) -> str:
        """Get PyTorch device string for current selection."""
        if self.selected_gpu_index == -1:
            return "cpu"
        elif TORCH_AVAILABLE and torch.cuda.is_available() and 0 <= self.selected_gpu_index < len(self.gpus):
            return f"cuda:{self.selected_gpu_index}"
        else:
            return "cpu"

    def get_onnx_providers(self) -> List[str]:
        """Get ONNX Runtime providers for current selection."""
        providers = []

        if self.selected_gpu_index >= 0 and ONNX_AVAILABLE:
            # Try CUDA provider for selected GPU
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': self.selected_gpu_index,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }))
            except Exception:
                pass

        # Always add CPU provider as fallback
        if ONNX_AVAILABLE:
            try:
                available_providers = ort.get_available_providers()
                if 'CPUExecutionProvider' in available_providers:
                    providers.append('CPUExecutionProvider')
            except Exception:
                pass

        return providers if providers else ['CPUExecutionProvider']

    def update_gpu_stats(self):
        """Update statistics for all GPUs."""
        for gpu in self.gpus:
            gpu.update_stats()

    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """Get GPU info by index."""
        if 0 <= index < len(self.gpus):
            return self.gpus[index]
        return None

    def get_selected_gpu_info(self) -> Optional[GPUInfo]:
        """Get info for currently selected GPU."""
        if 0 <= self.selected_gpu_index < len(self.gpus):
            return self.gpus[self.selected_gpu_index]
        return None

    def add_selection_callback(self, callback: callable):
        """Add callback to be called when GPU selection changes."""
        self.callbacks.append(callback)

    def remove_selection_callback(self, callback: callable):
        """Remove selection change callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _notify_callbacks(self):
        """Notify all callbacks of device selection change."""
        for callback in self.callbacks:
            try:
                callback(self.get_selected_device())
            except Exception as e:
                print(f"Error in GPU selection callback: {e}")

    def is_gpu_available(self) -> bool:
        """Check if any GPU is available."""
        return len(self.gpus) > 0

    def refresh_devices(self):
        """Refresh the list of available devices."""
        old_selection = self.selected_gpu_index
        self._detect_gpus()

        # Try to maintain selection if still valid
        if old_selection >= len(self.gpus):
            self.selected_gpu_index = 0 if self.gpus else -1
            self._notify_callbacks()


# Global instance
_gpu_service = None

def get_gpu_service() -> GPUService:
    """Get the global GPU service instance."""
    global _gpu_service
    if _gpu_service is None:
        _gpu_service = GPUService()
    return _gpu_service
