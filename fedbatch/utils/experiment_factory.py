
from glob import glob

from fedbatch.estimation.datasets import ExperimentDataset
from fedbatch.simulation.temperature_profile import TemperatureProfile
from fedbatch.simulation.induction_func import InductionProfile
from fedbatch.core.balances import FedBatchBalances
from fedbatch.simulation.feed_factory import create_feed
from fedbatch.core.fedbatch_model import FedBatchModel
from fedbatch.simulation.simulator import Simulator
from fedbatch.simulation.initial_conditions import build_initial_state
from fedbatch.utils.io import get_br_id

def build_experiments(cfg, kin):
    """
    Build datasets, simulators and initial conditions for all BR experiments.
    """
    dataset_files = sorted(glob("data/processed/BR*.xls"))
    datasets = [ExperimentDataset(f) for f in dataset_files]

    simulators = []
    y0s = []

    for dataset in datasets:
        br_id = get_br_id(dataset)  

        T_profile = TemperatureProfile(dataset.t, dataset.T)

        t_ind = cfg["bioreactor"][br_id]["t_ind"]["value"]
        I_profile = InductionProfile(t_ind)

        feed_cfg = cfg["feeds"][br_id]
        feed_S = create_feed(feed_cfg["feed_S"])
        feed_A = create_feed(feed_cfg["feed_A"])

        balances = FedBatchBalances(
            kinetics=kin,
            Sf=cfg["bioreactor"][br_id]["Sf"]["value"],
            temperature_profile=T_profile,
            induction_profile=I_profile
        )

        model = FedBatchModel(
            balances=balances,
            feed_S=feed_S,
            feed_A=feed_A
        )

        method = cfg["simulation"]["method_ode"]["type"]
        rtol = cfg["simulation"]["rtol_ode"]["value"]
        atol = cfg["simulation"]["atol_ode"]["value"]

        sim = Simulator(model, method, rtol, atol)
        simulators.append(sim)

        y0 = build_initial_state(cfg, br_id, dataset)
        y0s.append(y0)

    return datasets, simulators, y0s
