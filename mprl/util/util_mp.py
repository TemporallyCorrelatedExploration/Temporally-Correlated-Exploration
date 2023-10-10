from mp_pytorch.basis_gn import ProDMPBasisGenerator
from mp_pytorch.mp import ProDMP
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator

import mprl.util as util


def get_mp(**kwargs):
    assert kwargs["type"] == "prodmp"
    mp_args = kwargs["args"]
    dtype, device = util.parse_dtype_device(mp_args["dtype"], mp_args["device"])
    phase_gn = ExpDecayPhaseGenerator(tau=mp_args["tau"],
                                      delay=mp_args.get("delay", 0.0),
                                      alpha_phase=mp_args["alpha_phase"],
                                      learn_tau=mp_args.get("learn_tau", False),
                                      learn_delay=mp_args.get("learn_delay",
                                                              False),
                                      learn_alpha_phase=
                                      mp_args.get("learn_alpha_phase", False),
                                      dtype=dtype,
                                      device=device)
    basis_gn = ProDMPBasisGenerator(phase_generator=phase_gn,
                                    num_basis=mp_args["num_basis"],
                                    basis_bandwidth_factor=
                                    mp_args["basis_bandwidth_factor"],
                                    num_basis_outside=
                                    mp_args.get("num_basis_outside", 0),
                                    dt=mp_args["dt"],
                                    alpha=mp_args["alpha"],
                                    pre_compute_length_factor=5,
                                    dtype=dtype,
                                    device=device)
    prodmp = ProDMP(basis_gn=basis_gn,
                    num_dof=mp_args["num_dof"],
                    auto_scale_basis=mp_args.get("auto_scale_basis", True),
                    weights_scale=mp_args.get("weights_scale", 1),
                    goal_scale=mp_args.get("goal_scale", 1),
                    disable_weights=mp_args.get("disable_weights", False),
                    disable_goal=mp_args.get("disable_goal", False),
                    relative_goal=mp_args.get("relative_goal", False),
                    dtype=dtype,
                    device=device)
    return prodmp
