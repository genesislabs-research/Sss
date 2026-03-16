import torch
import os
from datetime import datetime
from typing import Dict, Any

class SerializationBridge:
    """
    Utility to bundle Isocortex, Allocortex, and Astrocytic states into 
    a single persistent .soul file.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def save_state(self, filepath: str = None):
        """
        Serializes the Total System Momentum.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"states/kernel_state_{timestamp}.soul"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Collect states from all subsystems.
        # The allocortex entry supports both the legacy AllocortexSystem (matrix/usage)
        # and the new HippocampalCore (hippocampal_core key). Both are written when
        # present so that checkpoints remain loadable during the transition period.
        allocortex_state = {}

        # HippocampalCore path (preferred: biologically parameterized formation)
        if hasattr(self.kernel.allocortex, "get_hippocampal_state"):
            allocortex_state["hippocampal_core"] = (
                self.kernel.allocortex.get_hippocampal_state()
            )

        # Legacy AllocortexSystem path (retained for backward compatibility)
        if hasattr(self.kernel.allocortex, "ca3") and hasattr(
            self.kernel.allocortex.ca3, "memory_matrix"
        ):
            allocortex_state["legacy_matrix"] = (
                self.kernel.allocortex.ca3.memory_matrix.cpu().clone()
            )
            allocortex_state["legacy_usage"] = (
                self.kernel.allocortex.ca3.usage_counters.cpu().clone()
            )

        state_package = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "allocortex_type": type(self.kernel.allocortex).__name__,
                "status": "stable_resume_point"
            },
            "isocortex": self.kernel.isocortex.get_serialized_state(),
            "allocortex": allocortex_state,
            "astrocyte": self.kernel.astrocyte.get_metabolic_state()
        }

        torch.save(state_package, filepath)
        print(f"Serialization Bridge: State successfully anchored to {filepath}")

    def resume_state(self, filepath: str):
        """
        Thaws the kernel and restores dynamical momentum.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No state found at {filepath}")

        package = torch.load(filepath)

        # Re-inject states into subsystems
        self.kernel.isocortex.set_serialized_state(package["isocortex"])
        self.kernel.astrocyte.set_metabolic_state(package["astrocyte"])

        # Restore allocortex — handles both HippocampalCore and legacy AllocortexSystem.
        allocortex_package = package["allocortex"]

        if "hippocampal_core" in allocortex_package and hasattr(
            self.kernel.allocortex, "set_hippocampal_state"
        ):
            # Preferred path: full biological formation state
            self.kernel.allocortex.set_hippocampal_state(
                allocortex_package["hippocampal_core"]
            )
        elif "legacy_matrix" in allocortex_package and hasattr(
            self.kernel.allocortex, "ca3"
        ):
            # Backward-compatible path: flat matrix/usage from old AllocortexSystem
            self.kernel.allocortex.ca3.memory_matrix.copy_(
                allocortex_package["legacy_matrix"]
            )
            self.kernel.allocortex.ca3.usage_counters.copy_(
                allocortex_package["legacy_usage"]
            )
        else:
            print("[SerializationBridge] WARNING: No recognizable allocortex state found "
                  "in checkpoint. Allocortex will initialize cold.")

        print(f"Serialization Bridge: Resume complete. Metadata: {package['metadata']}")
      
