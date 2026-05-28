"""Finishers — differentiable refinement stages for CRISP v0.4.

Finishers are short biased-relaxation runs inserted between GP filtering
and full unbiased relaxation. They steer candidates toward promising
regions of configuration space using FP-space objectives, then anneal
the bias to zero and finish with unbiased cleanup.

Available finishers:
- FPTargetFinisher: steer toward structural prototypes via per-atom FP matching
- gp_guided_relax: follow GP energy gradient in FP-space to discover new basins
"""

from .fp_target import FPTargetFinisher
from .gp_guided import GPGuidedConfig, KickRelaxConfig, gp_guided_relax, gp_kick_relax

__all__ = ["FPTargetFinisher", "GPGuidedConfig", "KickRelaxConfig",
           "gp_guided_relax", "gp_kick_relax"]
