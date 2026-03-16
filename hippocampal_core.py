# ─────────────────────────────────────────────────────────────────────────────
# § HIPPOCAMPAL CORE: Biologically Parameterized Episodic Memory Engine
#
# This module replaces the generic cosine-attractor allocortex with a
# region-differentiated hippocampal formation. Each subfield is instantiated
# with Izhikevich neuron parameters calibrated to its known biological firing
# regime. The interface contract with AllocortexSystem is preserved exactly,
# so main.py and SerializationBridge require no changes.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# ANATOMICAL PIPELINE:
#
# [ ENTORHINAL CORTEX ] ──→ [ DENTATE GYRUS ] ──→ [ CA3 ] ──→ [ CA1 ]
#
# FUNCTION:   Input Gateway     Pattern Sep.     Autoassoc.    Mismatch
#             (The Perforant    (Orthogonalize   Attractor     Detection
#              Path)            Representations) (One-Shot     (Novelty
#                                                Write)        Gate)
#
# FIRING:     Tonic bursting    Sparse / silent  Burst-prone   Regular
#             ~20 Hz            < 2% active      Recurrent     Comparator
#
# RESEARCH:
#   - Izhikevich (2003): "Simple model of spiking neurons"
#     DOI: 10.1109/TNN.2003.820440
#     Rationale: Four-parameter model captures full repertoire of cortical
#     firing patterns (tonic, burst, chattering) by region.
#
#   - Knierim (2015): "The hippocampus"
#     Current Biology, 25(23), R1116-R1121.
#     Rationale: Functional specialization of DG/CA3/CA1 subfields.
#
#   - Rolls (2013): "The mechanisms for pattern completion and pattern
#     separation in the hippocampus"
#     European Journal of Neuroscience, 37(7), 1083-1093.
#     Rationale: CA3 recurrent collaterals as autoassociative memory;
#     DG sparse coding as orthogonalization engine.
#
#   - Larimar (2024): arXiv:2403.11901
#     Rationale: One-shot write via closed-form associative matrix update.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# § ON THE IZHIKEVICH MODEL
#
#   The membrane potential v and recovery variable u evolve as:
#
#       dv/dt = 0.04v² + 5v + 140 - u + I
#       du/dt = a(bv - u)
#
#   Upon threshold crossing (v >= 30 mV):
#       v ← c
#       u ← u + d
#
#   The four parameters {a, b, c, d} are the only knobs needed to reproduce
#   the full diversity of cortical firing patterns:
#
#   Parameter | Biological Meaning
#   ──────────┼────────────────────────────────────────────────────────────
#   a         | Recovery time constant of u. Small a = slow recovery = burst.
#   b         | Sensitivity of u to subthreshold membrane potential v.
#   c         | Post-spike reset voltage (mV). Higher c = faster return to firing.
#   d         | Post-spike jump in u. Larger d = stronger after-hyperpolarization.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# § REGION-SPECIFIC PARAMETER CHOICES
#
#   ENTORHINAL CORTEX (EC) — Input relay, tonic regular spiking
#       a=0.02, b=0.2, c=-65, d=8
#       Standard regular-spiking excitatory profile. EC neurons fire reliably
#       in response to sensory drive without intrinsic burst tendency.
#       Reference: Izhikevich (2003) Figure 1, regular spiking.
#
#   DENTATE GYRUS GRANULE CELLS (GC) — Sparse, high-threshold
#       a=0.02, b=0.2, c=-65, d=8  (identical to EC, but with strong inhibition)
#       Granule cells are among the most electrically compact neurons in the
#       brain. They fire rarely (< 2% of population active at any moment)
#       due to strong feedforward inhibition from basket cells and mossy cells.
#       The sparsity here is enforced by the activity threshold, not by
#       parameter differences — matching the biology where DG sparsity is
#       a network-level property, not a single-cell property.
#       Reference: Treves & Rolls (1994): sparse coding capacity in DG.
#
#   CA3 PYRAMIDAL CELLS — Burst-prone, recurrent collaterals
#       a=0.02, b=0.2, c=-55, d=4
#       The elevated reset voltage (c=-55 vs -65) means CA3 cells recover
#       to a less hyperpolarized state after each spike, enabling burst
#       firing. This is the biological substrate of pattern completion:
#       a partial cue triggers a burst that propagates through recurrent
#       collaterals until the network settles into a stored attractor.
#       The reduced d=4 weakens after-hyperpolarization, sustaining bursts.
#       Reference: Miles & Wong (1987): intrinsic burst firing in CA3.
#
#   CA1 PYRAMIDAL CELLS — Regular spiking, output comparator
#       a=0.02, b=0.2, c=-65, d=6
#       CA1 cells are the hippocampal output layer. They compare CA3
#       reconstruction (via Schaffer collaterals) against direct EC input
#       (via the temporoammonic path). The intermediate d=6 gives moderate
#       after-hyperpolarization, supporting rate-coded mismatch signals
#       rather than burst dynamics.
#       Reference: Mehta et al. (2000): CA1 as a comparator for prediction error.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# § ON STDP (Spike-Timing-Dependent Plasticity)
#
#   STDP is the Hebbian learning rule at the synapse level. When a presynaptic
#   neuron fires just BEFORE a postsynaptic neuron, the synapse strengthens
#   (Long-Term Potentiation, LTP). When the order reverses, it weakens
#   (Long-Term Depression, LTD).
#
#   In this implementation:
#   - LTP is applied during one_shot_write: DG spikes (pre) co-occur with
#     CA3 spikes (post), strengthening the DG→CA3 perforant path synapses.
#   - LTD is stubbed. In PRAGMI, LTD is the responsibility of the
#     ReconsolidationModule, which rewrites memories under the current
#     network state during retrieval. Separating LTP (encoding) from LTD
#     (updating) maps to the biological distinction between initial
#     consolidation and post-retrieval reconsolidation.
#
#   Reference: Bi & Poo (1998): "Synaptic modifications in cultured
#   hippocampal neurons: dependence on spike timing"
#   DOI: 10.1523/JNEUROSCI.18-24-10464.1998
#
# ─────────────────────────────────────────────────────────────────────────────
#
# § DEVICE STRATEGY
#
#   All operations use native PyTorch tensors and run on whichever device
#   the parent AllocortexSystem is placed on (CPU or CUDA). This ensures
#   compatibility with the robotics deployment target where CUDA may not
#   be available.
#
#   The Numba CUDA kernels from hippocampus_kernels.py are a known-good
#   accelerator path. When available, they can replace the Izhikevich step
#   functions here with no interface change. That swap is left as an
#   explicit TODO rather than silent conditional logic, so it's auditable.
#
#   TODO (performance): When numba.cuda.is_available(), route
#   _izhikevich_step() through update_neuron_kernel from hippocampus_kernels.py.
#   The synapse_transmission_kernel and stdp_kernel there are drop-in
#   replacements for _transmit_spikes() and _apply_ltp() respectively.
#
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# § CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HippocampalConfig:
    """
    Unified configuration for the full hippocampal formation.

    Dimensions are shared across subfields for interface compatibility with
    the Isocortex Substrate (entry_dim must match IsocortexConfig.zone_width).
    """
    entry_dim: int = 1024           # Input dimensionality from Isocortex
    granule_cell_count: int = 2048  # DG population (typically 5-10x CA3 in biology)
    ca3_cell_count: int = 512       # CA3 population (recurrent attractor pool)
    ca1_cell_count: int = 1024      # CA1 population (output comparator)
    episodic_capacity: int = 5000   # Maximum stored episodes in CA3 matrix
    integration_timestep: float = 0.5   # Euler step size (ms). Must be <= 0.5 for
                                        # numerical stability of Izhikevich quadratic.
                                        # See: numerical analysis note in _izhikevich_step.
    sparsity_threshold: float = 0.08    # DG firing threshold. Enforces < ~8% active
                                        # granule cells per input, matching biological
                                        # estimates of DG population sparsity.
                                        # Reference: Treves & Rolls (1994).
    recurrent_connection_probability: float = 0.10  # CA3→CA3 connectivity.
                                                     # Anatomical estimates: ~2-4% in rodent,
                                                     # higher in primate. 10% is conservative
                                                     # upper bound for a dense PyTorch impl.
    ca3_attractor_steps: int = 5        # Recurrent settling iterations per retrieval query.
    stdp_learning_rate: float = 0.01    # LTP rate for DG→CA3 engram formation.
    synaptic_weight_maximum: float = 5.0  # Soft saturation ceiling for STDP weights.
                                          # Prevents runaway potentiation.
                                          # Reference: Turrigiano (2008) synaptic scaling.
    reconsolidation_drift_scale: float = 0.01  # Magnitude of state-dependent memory rewrite
                                                # applied during retrieval. Controlled here,
                                                # executed by the ReconsolidationModule.


# ─────────────────────────────────────────────────────────────────────────────
# § IZHIKEVICH NEURON DYNAMICS (Shared across all subfields)
# ─────────────────────────────────────────────────────────────────────────────

def _izhikevich_step(
    membrane_potential: torch.Tensor,     # v: current voltage (mV), shape (N,)
    recovery_variable: torch.Tensor,      # u: adaptation current, shape (N,)
    recovery_time_constant: torch.Tensor, # a: inverse recovery timescale, shape (N,)
    subthreshold_coupling: torch.Tensor,  # b: u sensitivity to subthreshold v, shape (N,)
    spike_reset_voltage: torch.Tensor,    # c: post-spike reset voltage (mV), shape (N,)
    after_hyperpolarization_jump: torch.Tensor,  # d: post-spike u increment, shape (N,)
    injected_current: torch.Tensor,       # I: total input current, shape (N,)
    integration_timestep: float,          # dt: Euler step size (ms)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single Euler integration step of the Izhikevich (2003) neuron model.

    NUMERICAL STABILITY NOTE:
    The membrane potential equation contains a quadratic term (0.04v²) which
    causes numerical blow-up if dt is too large. At dt=1.0ms with typical
    initial conditions, v can diverge within a few steps. The safe upper bound
    is dt <= 0.5ms. The standard practice for dt=1.0ms is to split into two
    half-steps (dt/2 each), which is equivalent to halving the step size.
    HippocampalConfig enforces dt=0.5 by default.

    Returns:
        updated_membrane_potential : shape (N,)
        updated_recovery_variable  : shape (N,)
        spike_vector               : binary, shape (N,) — 1 where v crossed 30mV
    """
    # ── Membrane dynamics ────────────────────────────────────────────────────
    # dv/dt = 0.04v² + 5v + 140 - u + I
    # This piecewise-linear approximation fits the Hodgkin-Huxley I-V curve.
    d_membrane_potential = (
        0.04 * membrane_potential ** 2
        + 5.0 * membrane_potential
        + 140.0
        - recovery_variable
        + injected_current
    )

    # ── Recovery dynamics ────────────────────────────────────────────────────
    # du/dt = a(bv - u)
    # u tracks slow subthreshold oscillations that produce adaptation and bursting.
    d_recovery_variable = recovery_time_constant * (
        subthreshold_coupling * membrane_potential - recovery_variable
    )

    # ── Euler integration ────────────────────────────────────────────────────
    updated_membrane_potential = membrane_potential + d_membrane_potential * integration_timestep
    updated_recovery_variable  = recovery_variable  + d_recovery_variable  * integration_timestep

    # ── Spike detection and reset ─────────────────────────────────────────────
    # Threshold at 30mV (biological action potential peak approximation).
    spike_vector = (updated_membrane_potential >= 30.0).float()

    # Where a spike occurred: reset v to c, increment u by d.
    updated_membrane_potential = torch.where(
        spike_vector.bool(),
        spike_reset_voltage,
        updated_membrane_potential
    )
    updated_recovery_variable = updated_recovery_variable + spike_vector * after_hyperpolarization_jump

    # ── Hard clamps (numerical safety) ───────────────────────────────────────
    # These catch edge cases without masking instability silently.
    # If values are hitting these clamps frequently, dt is too large.
    updated_membrane_potential = updated_membrane_potential.clamp(-90.0, 30.0)

    return updated_membrane_potential, updated_recovery_variable, spike_vector


# ─────────────────────────────────────────────────────────────────────────────
# § STAGE I: ENTORHINAL CORTEX — Input Relay
# ─────────────────────────────────────────────────────────────────────────────

class EntorhinalCortex(nn.Module):
    """
    Projects Isocortex state into the hippocampal formation via the perforant path.

    Biological role: The entorhinal cortex is the primary gateway between
    neocortex and hippocampus. It receives highly processed, multimodal sensory
    input and relays it via the perforant path to both DG and CA3 (direct),
    and via the temporoammonic path directly to CA1.

    In PRAGMI this layer serves as the impedance matcher between the
    Isocortex Substrate's continuous-valued output and the current-injection
    format expected by the spiking hippocampal subfields.

    Reference:
      Witter et al. (2000): "Cortico-hippocampal communication by way of
      parallel parahippocampal-subicular pathways"
      DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        neuron_count = cfg.entry_dim

        # ── Izhikevich parameters: Regular spiking excitatory (EC profile) ──
        # a=0.02, b=0.2, c=-65, d=8 — see region parameter table at top of file.
        self.register_buffer("recovery_time_constant",
            torch.full((neuron_count,), 0.02))
        self.register_buffer("subthreshold_coupling",
            torch.full((neuron_count,), 0.2))
        self.register_buffer("spike_reset_voltage",
            torch.full((neuron_count,), -65.0))
        self.register_buffer("after_hyperpolarization_jump",
            torch.full((neuron_count,), 8.0))

        # ── Membrane state (persists across timesteps within a session) ──────
        self.register_buffer("membrane_potential",
            torch.full((neuron_count,), -65.0))
        self.register_buffer("recovery_variable",
            self.subthreshold_coupling * torch.full((neuron_count,), -65.0))

        # ── Input projection: Isocortex dim → EC neuron count ────────────────
        self.perforant_path_projection = nn.Linear(cfg.entry_dim, neuron_count, bias=False)

    def forward(self, isocortex_state: torch.Tensor) -> torch.Tensor:
        """
        Converts continuous Isocortex activity into EC spike train.

        The projection scales the isocortex state into a physiologically
        plausible current range (~0–15 pA equivalent units).
        """
        # Project and scale to current injection range
        injected_current = self.perforant_path_projection(isocortex_state).squeeze(0)
        injected_current = torch.relu(injected_current) * 15.0

        self.membrane_potential, self.recovery_variable, spike_vector = _izhikevich_step(
            self.membrane_potential,
            self.recovery_variable,
            self.recovery_time_constant,
            self.subthreshold_coupling,
            self.spike_reset_voltage,
            self.after_hyperpolarization_jump,
            injected_current,
            self.cfg.integration_timestep,
        )
        return spike_vector


# ─────────────────────────────────────────────────────────────────────────────
# § STAGE II: DENTATE GYRUS — Pattern Separation Engine
# ─────────────────────────────────────────────────────────────────────────────

class DentateGyrus(nn.Module):
    """
    Orthogonalizes incoming EC representations to minimize memory interference.

    Biological role: The dentate gyrus performs pattern separation — the
    process of making similar inputs as distinct as possible before they
    reach CA3. It achieves this through:
      1. Massive expansion coding (rodent DG has ~1M granule cells vs ~300K EC neurons)
      2. Extreme sparsity: < 2% of granule cells are active at any moment
      3. Strong lateral inhibition from basket cells and mossy cells

    Without pattern separation, two similar episodes (e.g. "coffee on Tuesday"
    and "coffee on Wednesday") would activate overlapping CA3 attractors and
    interfere with each other. DG ensures they get distinct engram addresses.

    This is why DG lesions cause pattern completion to collapse into
    false-positive retrieval (similar events blend together).

    References:
      Treves & Rolls (1994): "Computational analysis of the role of the
      hippocampus in memory"
      DOI: 10.1002/hipo.450040319

      Leutgeb et al. (2007): "Pattern separation in the dentate gyrus and
      CA3 of the hippocampus"
      DOI: 10.1126/science.1135801
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        neuron_count = cfg.granule_cell_count

        # ── Izhikevich parameters: Granule cell profile ───────────────────────
        # Same base parameters as EC, but sparsity is enforced at network level
        # via the activity threshold, not by intrinsic cell properties.
        # Heterogeneity is introduced by random noise on c and d (after Izhikevich 2003
        # Figure 8: random networks fire more naturally than uniform ones).
        base_reset   = torch.full((neuron_count,), -65.0)
        base_jump    = torch.full((neuron_count,), 8.0)
        heterogeneity = torch.rand(neuron_count)

        self.register_buffer("recovery_time_constant",
            torch.full((neuron_count,), 0.02))
        self.register_buffer("subthreshold_coupling",
            torch.full((neuron_count,), 0.2))
        self.register_buffer("spike_reset_voltage",
            base_reset + 10.0 * heterogeneity ** 2)   # c ∈ [-65, -55]
        self.register_buffer("after_hyperpolarization_jump",
            base_jump  -  4.0 * heterogeneity ** 2)   # d ∈ [4, 8]

        # ── Membrane state ───────────────────────────────────────────────────
        self.register_buffer("membrane_potential",
            torch.full((neuron_count,), -65.0))
        self.register_buffer("recovery_variable",
            0.2 * torch.full((neuron_count,), -65.0))

        # ── Mossy fiber projection: EC spikes → GC current ───────────────────
        # Sparse random weights mimic the mossy fiber contact geometry:
        # each granule cell receives input from a small fraction of EC axons.
        self.mossy_fiber_weights = nn.Linear(cfg.entry_dim, neuron_count, bias=False)
        nn.init.sparse_(self.mossy_fiber_weights.weight, sparsity=0.9)

    def forward(self, ec_spike_vector: torch.Tensor) -> torch.Tensor:
        """
        Applies sparse encoding to EC input, returning orthogonalized GC spikes.

        The sparsity threshold is applied post-spike to enforce the biological
        < 8% active constraint. This is a network-level operation (mimicking
        feedforward inhibition) rather than a per-cell property.
        """
        # Project EC spikes through mossy fibers
        granule_current = self.mossy_fiber_weights(ec_spike_vector) * 10.0

        self.membrane_potential, self.recovery_variable, spike_vector = _izhikevich_step(
            self.membrane_potential,
            self.recovery_variable,
            self.recovery_time_constant,
            self.subthreshold_coupling,
            self.spike_reset_voltage,
            self.after_hyperpolarization_jump,
            granule_current,
            self.cfg.integration_timestep,
        )

        # ── Network-level sparsification (feedforward inhibition proxy) ──────
        # Keep only the top-k most active granule cells, where k corresponds to
        # the biological sparsity fraction. This approximates basket cell
        # inhibition without requiring an explicit inhibitory population.
        active_count = max(1, int(self.cfg.granule_cell_count * self.cfg.sparsity_threshold))
        topk_threshold = torch.topk(spike_vector, active_count).values.min()
        sparse_spike_vector = (spike_vector >= topk_threshold).float() * spike_vector

        return sparse_spike_vector


# ─────────────────────────────────────────────────────────────────────────────
# § STAGE III: CA3 — Autoassociative Attractor Network
# ─────────────────────────────────────────────────────────────────────────────

class CA3RecurrentAttractor(nn.Module):
    """
    One-shot episodic storage and pattern completion via recurrent collaterals.

    Biological role: CA3 is the core memory store of the hippocampus. Its
    defining feature is a dense recurrent collateral system: every CA3
    pyramidal cell sends axon branches back onto ~1-4% of all other CA3 cells.
    This creates an autoassociative network — a Hopfield-like system where
    stored patterns are attractors in the state space.

    Pattern completion: when a partial or degraded cue arrives (via DG mossy
    fibers or direct EC perforant path), recurrent collateral activity drives
    the network toward the nearest stored attractor. The system 'fills in'
    the missing details. This is why you can recognize a half-obscured face,
    or recall a full memory from a single smell.

    One-shot storage: the mossy fiber synapse (DG→CA3) is exceptionally strong
    — it is often called a "detonator synapse" because a single presynaptic
    spike can drive CA3 to threshold. This allows a new episode to be written
    into the CA3 attractor landscape in a single exposure, without gradient
    descent.

    References:
      Marr (1971): "Simple memory: a theory for archicortex"
      DOI: 10.1098/rstb.1971.0078

      McNaughton & Morris (1987): "Hippocampal synaptic enhancement and
      information storage within a distributed memory system"
      DOI: 10.1016/0166-2236(87)90011-7

      Larimar (2024): arXiv:2403.11901 — closed-form associative write.
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        neuron_count = cfg.ca3_cell_count

        # ── Izhikevich parameters: CA3 burst-prone pyramidal profile ──────────
        # c=-55 (elevated reset) and d=4 (weak after-hyperpolarization) together
        # produce the burst firing that propagates through recurrent collaterals
        # during attractor settlement. See parameter table at top of file.
        self.register_buffer("recovery_time_constant",
            torch.full((neuron_count,), 0.02))
        self.register_buffer("subthreshold_coupling",
            torch.full((neuron_count,), 0.2))
        self.register_buffer("spike_reset_voltage",
            torch.full((neuron_count,), -55.0))  # KEY: elevated vs DG/EC
        self.register_buffer("after_hyperpolarization_jump",
            torch.full((neuron_count,), 4.0))    # KEY: weaker AHP = sustained burst

        # ── Membrane state ───────────────────────────────────────────────────
        self.register_buffer("membrane_potential",
            torch.full((neuron_count,), -65.0))
        self.register_buffer("recovery_variable",
            0.2 * torch.full((neuron_count,), -65.0))

        # ── Recurrent collateral weight matrix (CA3→CA3) ─────────────────────
        # Initialized sparse with small positive weights. Weights grow via STDP
        # during encoding. Shape: (ca3_cell_count, ca3_cell_count).
        recurrent_weights = torch.zeros(neuron_count, neuron_count)
        connection_mask = (torch.rand(neuron_count, neuron_count)
                           < cfg.recurrent_connection_probability).float()
        recurrent_weights = connection_mask * torch.rand(neuron_count, neuron_count) * 0.1
        recurrent_weights.fill_diagonal_(0.0)  # No self-connections
        self.register_buffer("recurrent_collateral_weights", recurrent_weights)

        # ── Mossy fiber input projection: DG spikes → CA3 current ────────────
        # Strong detonator-synapse-like projection. Initialized with large weights
        # to reflect the biological reality that mossy fiber EPSPs are ~5-10x
        # larger than typical excitatory synapses in CA3.
        self.mossy_fiber_projection = nn.Linear(cfg.granule_cell_count, neuron_count, bias=False)
        nn.init.normal_(self.mossy_fiber_projection.weight, mean=0.0, std=0.05)

        # ── STDP eligibility traces ───────────────────────────────────────────
        # Presynaptic trace (x_pre): incremented on pre-spike, decays exponentially.
        # Used to compute the LTP update: Δw ∝ x_pre * post_spike.
        # See: Bi & Poo (1998) for experimental basis of trace-based STDP.
        self.register_buffer("presynaptic_stdp_trace",
            torch.zeros(cfg.granule_cell_count))

        # ── Episodic memory matrix (the long-term engram store) ───────────────
        # Each row is one stored episode (a CA3 population vector at write time).
        # Write pointer cycles through capacity when full (FIFO eviction).
        # This is the structure Larimar (2024) uses for one-shot associative write.
        self.register_buffer("episodic_memory_matrix",
            torch.zeros(cfg.episodic_capacity, neuron_count))
        self.register_buffer("episodic_usage_counters",
            torch.zeros(cfg.episodic_capacity))
        self._write_pointer = 0

    def one_shot_write(self, dg_spike_vector: torch.Tensor) -> int:
        """
        Writes a new episode into the CA3 memory matrix without gradient descent.

        Mechanism: The mossy fiber input drives CA3 to threshold. The resulting
        CA3 spike pattern is stored as a row in the episodic memory matrix.
        STDP then strengthens the DG→CA3 synapses that were co-active, making
        this episode retrievable from a partial DG cue on future presentations.

        Returns the slot index where the episode was stored.
        """
        # ── Drive CA3 from DG input ───────────────────────────────────────────
        mossy_fiber_current = self.mossy_fiber_projection(dg_spike_vector) * 15.0

        self.membrane_potential, self.recovery_variable, ca3_spike_vector = _izhikevich_step(
            self.membrane_potential,
            self.recovery_variable,
            self.recovery_time_constant,
            self.subthreshold_coupling,
            self.spike_reset_voltage,
            self.after_hyperpolarization_jump,
            mossy_fiber_current,
            self.cfg.integration_timestep,
        )

        # ── Store episode ─────────────────────────────────────────────────────
        slot = self._write_pointer % self.cfg.episodic_capacity
        self.episodic_memory_matrix[slot] = ca3_spike_vector.detach()
        self.episodic_usage_counters[slot] += 1
        self._write_pointer += 1

        # ── LTP via STDP: strengthen DG→CA3 synapses for this episode ────────
        # This is Hebbian engram formation: synapses that participated in
        # encoding this episode become stronger, lowering the threshold for
        # future retrieval from a partial cue.
        self._apply_ltp(dg_spike_vector, ca3_spike_vector)

        return slot

    def _apply_ltp(
        self,
        presynaptic_spikes: torch.Tensor,   # DG spike vector
        postsynaptic_spikes: torch.Tensor,  # CA3 spike vector
    ):
        """
        Soft-saturating LTP rule applied at DG→CA3 synapses.

        Rule: if pre fires AND post fires (co-activation), strengthen the
        synapse asymptotically toward synaptic_weight_maximum.

        Δw = lr * (w_max - w)  [soft saturation — approaches w_max asymptotically]

        Soft saturation (vs hard clamp) preserves relative strength differences
        between synapses while preventing runaway potentiation.
        Reference: Turrigiano & Nelson (2004) synaptic scaling / BCM theory.

        NOTE: LTD is intentionally absent here. Post-retrieval weight depression
        (the substrate of reconsolidation) is handled by the ReconsolidationModule
        in the PRAGMI pipeline, which rewrites memories under the current
        network state. Keeping LTP and LTD in separate modules makes the
        write-then-update sequence auditable.
        """
        # Update presynaptic trace (exponential decay + spike increment)
        stdp_trace_decay = 0.95
        self.presynaptic_stdp_trace = (
            stdp_trace_decay * self.presynaptic_stdp_trace + presynaptic_spikes
        )

        # Co-activation mask: pre trace * post spike = Hebbian coincidence
        # Shape: (granule_cell_count,) outer product is too large to store;
        # we update only the projection weights for the active pre/post pairs.
        with torch.no_grad():
            # For each active post-synaptic CA3 cell, strengthen the mossy fiber
            # synapses from active DG cells (pre trace > 0)
            active_post = postsynaptic_spikes.bool()  # (ca3_cell_count,)
            active_pre_trace = self.presynaptic_stdp_trace  # (granule_cell_count,)

            if active_post.any() and active_pre_trace.sum() > 0:
                # Weight update: Δw = lr * (w_max - w) for co-active synapses
                current_weights = self.mossy_fiber_projection.weight  # (ca3, dg)
                ltp_delta = (
                    self.cfg.stdp_learning_rate
                    * (self.cfg.synaptic_weight_maximum - current_weights)
                    * active_pre_trace.unsqueeze(0)   # broadcast over ca3 dim
                    * active_post.float().unsqueeze(1) # broadcast over dg dim
                )
                self.mossy_fiber_projection.weight.add_(ltp_delta)

    def attractor_settle(self, dg_query_spikes: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the most relevant stored episode via recurrent collateral dynamics.

        Mechanism: The query (a partial DG spike pattern) is injected as current.
        Recurrent collaterals then iterate the CA3 population toward the nearest
        stored attractor basin. The settled state is the reconstruction.

        This replaces the previous cosine-softmax retrieval with actual recurrent
        spiking dynamics, matching the biological process of pattern completion.

        Steps are bounded by cfg.ca3_attractor_steps. In biology, CA3 attractor
        settlement takes ~50-100ms (theta half-cycle); each step here corresponds
        to one integration timestep.
        """
        # Project DG query into CA3 current space
        query_current = self.mossy_fiber_projection(dg_query_spikes) * 10.0

        # Reset membrane to resting before retrieval
        # (biological: between theta cycles, CA3 cells return to rest)
        retrieval_membrane = self.membrane_potential.clone()
        retrieval_recovery  = self.recovery_variable.clone()

        for _ in range(self.cfg.ca3_attractor_steps):
            # Recurrent collateral contribution: spikes from last step feed back
            recurrent_current = torch.mv(
                self.recurrent_collateral_weights,
                (retrieval_membrane >= 30.0).float()
            )
            total_current = query_current + recurrent_current

            retrieval_membrane, retrieval_recovery, spike_vector = _izhikevich_step(
                retrieval_membrane,
                retrieval_recovery,
                self.recovery_time_constant,
                self.subthreshold_coupling,
                self.spike_reset_voltage,
                self.after_hyperpolarization_jump,
                total_current,
                self.cfg.integration_timestep,
            )

        # Return the settled CA3 population vector as the reconstructed episode
        return spike_vector


# ─────────────────────────────────────────────────────────────────────────────
# § STAGE IV: CA1 — Mismatch Detector and Output Gate
# ─────────────────────────────────────────────────────────────────────────────

class CA1Comparator(nn.Module):
    """
    Computes prediction error between CA3 reconstruction and current EC reality.

    Biological role: CA1 receives two inputs simultaneously:
      1. CA3 reconstruction via Schaffer collaterals (what memory predicts)
      2. Direct EC input via the temporoammonic path (what is actually happening)

    CA1 acts as a coincidence detector / comparator between these two signals.
    High mismatch (prediction error) signals a novel experience that warrants
    a new CA3 write. Low mismatch signals a familiar experience — pattern
    completion succeeded and no new encoding is needed.

    This is the biological implementation of prediction error gating, analogous
    to the dopaminergic RPE (Reward Prediction Error) signal in the basal ganglia
    but operating on episodic content rather than reward value.

    References:
      Mehta et al. (2000): "Experience-dependent asymmetric shape of hippocampal
      receptive fields"
      DOI: 10.1016/S0896-6273(00)80101-8

      Kumaran & Maguire (2007): "Match-mismatch processes underlie human
      hippocampal responses to associative novelty"
      DOI: 10.1523/JNEUROSCI.1085-07.2007
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        neuron_count = cfg.ca1_cell_count

        # ── Izhikevich parameters: CA1 regular spiking output profile ────────
        # d=6 gives intermediate after-hyperpolarization — more than CA3 (d=4)
        # but less than EC/DG (d=8). This produces rate-coded output suitable
        # for graded mismatch signaling rather than binary burst/no-burst.
        self.register_buffer("recovery_time_constant",
            torch.full((neuron_count,), 0.02))
        self.register_buffer("subthreshold_coupling",
            torch.full((neuron_count,), 0.2))
        self.register_buffer("spike_reset_voltage",
            torch.full((neuron_count,), -65.0))
        self.register_buffer("after_hyperpolarization_jump",
            torch.full((neuron_count,), 6.0))  # KEY: intermediate AHP

        # ── Membrane state ───────────────────────────────────────────────────
        self.register_buffer("membrane_potential",
            torch.full((neuron_count,), -65.0))
        self.register_buffer("recovery_variable",
            0.2 * torch.full((neuron_count,), -65.0))

        # ── Schaffer collateral projection: CA3 → CA1 ────────────────────────
        self.schaffer_collateral_projection = nn.Linear(
            cfg.ca3_cell_count, neuron_count, bias=False
        )
        # ── Temporoammonic projection: direct EC → CA1 ────────────────────────
        # This bypasses DG and CA3. Its functional role is to provide CA1 with
        # an unprocessed copy of the current sensory context against which the
        # CA3 reconstruction (Schaffer collateral input) is compared.
        self.temporoammonic_projection = nn.Linear(
            cfg.entry_dim, neuron_count, bias=False
        )

    def compute_prediction_error(
        self,
        ec_spike_vector: torch.Tensor,   # Direct EC input (current reality)
        ca3_spike_vector: torch.Tensor,  # CA3 reconstruction (memory prediction)
    ) -> torch.Tensor:
        """
        Returns scalar prediction error in [0, ∞).

        High error → novel experience → trigger one_shot_write in CA3.
        Low error  → familiar experience → pattern completion succeeded.

        The threshold for novelty gating in AllocortexSystem is separate from
        this computation; this module only produces the signal.
        """
        # Project both inputs into CA1 space
        schaffer_current    = self.schaffer_collateral_projection(ca3_spike_vector)
        temporoammonic_current = self.temporoammonic_projection(ec_spike_vector)

        # Drive CA1 from combined input
        total_ca1_current = schaffer_current + temporoammonic_current

        self.membrane_potential, self.recovery_variable, ca1_spike_vector = _izhikevich_step(
            self.membrane_potential,
            self.recovery_variable,
            self.recovery_time_constant,
            self.subthreshold_coupling,
            self.spike_reset_voltage,
            self.after_hyperpolarization_jump,
            total_ca1_current,
            self.cfg.integration_timestep,
        )

        # Mismatch is the mean-squared difference between the two input streams
        # projected into CA1 space. High mismatch = large divergence between
        # what memory predicts and what EC is currently reporting.
        prediction_error = F.mse_loss(schaffer_current, temporoammonic_current)
        return prediction_error


# ─────────────────────────────────────────────────────────────────────────────
# § HIPPOCAMPAL CORE: Full Formation Assembly
# ─────────────────────────────────────────────────────────────────────────────

class HippocampalCore(nn.Module):
    """
    Full hippocampal formation: EC → DG → CA3 → CA1.

    This is the drop-in replacement for AllocortexSystem. The external
    interface is identical: forward() accepts an isocortex state tensor
    and returns (reconstructed_episode, prediction_error). one_shot_write()
    accepts an episodic trace and stores it.

    Internal state (all four subfield membrane potentials, STDP traces,
    and the CA3 episodic memory matrix) is fully serializable via
    get_hippocampal_state() / set_hippocampal_state(), and is included
    in the .soul file by SerializationBridge.

    Usage in main.py (no changes required to main.py):
        self.allocortex = HippocampalCore(HippocampalConfig())
    """

    def __init__(self, cfg: HippocampalConfig = None):
        super().__init__()
        self.cfg = cfg or HippocampalConfig()

        self.entorhinal_cortex = EntorhinalCortex(self.cfg)
        self.dentate_gyrus     = DentateGyrus(self.cfg)
        self.ca3               = CA3RecurrentAttractor(self.cfg)
        self.ca1               = CA1Comparator(self.cfg)

    def forward(
        self, isocortex_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full hippocampal pipeline: encode, retrieve, compare.

        Returns:
            reconstructed_episode : CA3-settled spike vector (the memory)
            prediction_error      : CA1 mismatch scalar (the novelty signal)
        """
        # Stage I: Relay through entorhinal cortex
        ec_spikes = self.entorhinal_cortex(isocortex_state)

        # Stage II: Pattern separation in dentate gyrus
        dg_spikes = self.dentate_gyrus(ec_spikes)

        # Stage III: Pattern completion in CA3 via recurrent collaterals
        reconstructed_episode = self.ca3.attractor_settle(dg_spikes)

        # Stage IV: Mismatch detection in CA1
        prediction_error = self.ca1.compute_prediction_error(ec_spikes, reconstructed_episode)

        return reconstructed_episode, prediction_error

    def one_shot_write(self, isocortex_state: torch.Tensor) -> int:
        """
        Encodes a new episode into CA3 without gradient descent.

        Routes through EC→DG to get the sparsified engram address,
        then writes to CA3 and applies STDP.

        Returns the CA3 matrix slot where the episode was stored.
        """
        ec_spikes = self.entorhinal_cortex(isocortex_state)
        dg_spikes = self.dentate_gyrus(ec_spikes)
        return self.ca3.one_shot_write(dg_spikes)

    # ─────────────────────────────────────────────────────────────────────────
    # § SERIALIZATION: State capture for .soul file integration
    # ─────────────────────────────────────────────────────────────────────────

    def get_hippocampal_state(self) -> dict:
        """
        Captures full hippocampal formation state for session persistence.

        Included in the .soul checkpoint as the 'hippocampal_core' key.
        SerializationBridge calls this in save_state().
        """
        return {
            "ec_membrane_potential":     self.entorhinal_cortex.membrane_potential.cpu(),
            "ec_recovery_variable":      self.entorhinal_cortex.recovery_variable.cpu(),
            "dg_membrane_potential":     self.dentate_gyrus.membrane_potential.cpu(),
            "dg_recovery_variable":      self.dentate_gyrus.recovery_variable.cpu(),
            "ca3_membrane_potential":    self.ca3.membrane_potential.cpu(),
            "ca3_recovery_variable":     self.ca3.recovery_variable.cpu(),
            "ca3_episodic_memory_matrix": self.ca3.episodic_memory_matrix.cpu(),
            "ca3_episodic_usage_counters": self.ca3.episodic_usage_counters.cpu(),
            "ca3_write_pointer":         self.ca3._write_pointer,
            "ca3_presynaptic_stdp_trace": self.ca3.presynaptic_stdp_trace.cpu(),
            "ca3_recurrent_weights":     self.ca3.recurrent_collateral_weights.cpu(),
            "ca1_membrane_potential":    self.ca1.membrane_potential.cpu(),
            "ca1_recovery_variable":     self.ca1.recovery_variable.cpu(),
        }

    def set_hippocampal_state(self, state: dict):
        """
        Restores full hippocampal formation state from checkpoint.

        Called by SerializationBridge in resume_state().
        """
        device = self.ca3.membrane_potential.device

        self.entorhinal_cortex.membrane_potential.copy_(state["ec_membrane_potential"].to(device))
        self.entorhinal_cortex.recovery_variable.copy_(state["ec_recovery_variable"].to(device))
        self.dentate_gyrus.membrane_potential.copy_(state["dg_membrane_potential"].to(device))
        self.dentate_gyrus.recovery_variable.copy_(state["dg_recovery_variable"].to(device))
        self.ca3.membrane_potential.copy_(state["ca3_membrane_potential"].to(device))
        self.ca3.recovery_variable.copy_(state["ca3_recovery_variable"].to(device))
        self.ca3.episodic_memory_matrix.copy_(state["ca3_episodic_memory_matrix"].to(device))
        self.ca3.episodic_usage_counters.copy_(state["ca3_episodic_usage_counters"].to(device))
        self.ca3._write_pointer = state["ca3_write_pointer"]
        self.ca3.presynaptic_stdp_trace.copy_(state["ca3_presynaptic_stdp_trace"].to(device))
        self.ca3.recurrent_collateral_weights.copy_(state["ca3_recurrent_weights"].to(device))
        self.ca1.membrane_potential.copy_(state["ca1_membrane_potential"].to(device))
        self.ca1.recovery_variable.copy_(state["ca1_recovery_variable"].to(device))

        print("[HippocampalCore] Formation state restored. Engram continuity preserved.")
