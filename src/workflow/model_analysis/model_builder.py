
from pathlib import Path
from src.utils.io import load_yaml

class ModelBuilderWorkflow:
    def __init__(self,instructions_path="src/config/model.yaml",operation_mode="FedBatch"):
        self.path = instructions_path
        self.operation_mode = operation_mode

    def load(self):
        self.model_ins = load_yaml(self.path)["model_instructions"]

    def constructor(self):
        if self.operation_mode == "FedBatch":
            code = self._build_fedbatch_odes()
        else:
            raise ValueError(f"Operation mode '{self.operation_mode}' not implemented.")
        output_file = Path("src/model/core/odes.py")
        with open(output_file, "w") as f:f.write(code)
        print(f"Model generated: {output_file}")

    def run(self):
        print("Loading instructions...")
        self.load()

        print("Generating model...")
        self.constructor()

    # Private methods

    # === FEDBATCH ODE GENERATOR ===
    def _build_fedbatch_odes(self):
        volume = self.model_ins["volume"]
        states = self.model_ins["states"]
        code = []

        # ------ Imports ----------
        code.append("import numpy as np")
        code.append("")
        code.append("from src.utils.io import load_yaml")
        code.append("from src.utils.imports import import_function")
        code.append("")

        # --------- Function definition ------------
        code.append("def ODEs(self, t, state):")
        code.append("")

        # ------ States unpacking ----------
        state_names = list(states.keys())
        code.append(f"    {', '.join(state_names + [volume['name']])} = state")
        code.append("")
        code.append("    states = {")
        for s in state_names:
            code.append(f"        '{s}': {s},")
        code.append(f"        '{volume['name']}': {volume['name']}")
        code.append("    }")
        code.append("")
        
        # --------- Parameters --------
        code.extend(self._build_parameters())

        # -------------- Inputs ------------
        code.extend(self._build_inputs())

        # -------------- Features ------------
        code.extend(self._build_features())

        # --------- Volume Balance ----------
        code.extend(self._build_volume_balance(volume))

        # ----------- Kinetics ------------------------
        code.extend(self._build_kinetic_functions())

        # -------- State balances -----------------
        for state_name, state_cfg in states.items():
            code.extend(self._build_state_balance(state_name,state_cfg,volume))

        # --------- Return ----------------------
        derivatives = [f"d{s}dt" for s in state_names]
        derivatives.append(f"d{volume['name']}dt")
        code.append("")
        code.append(f"    return np.array([{', '.join(derivatives)}])")

        return "\n".join(code)

    # =============== PARAMETERS ====================
    def _build_parameters(self):
        path = (self.model_ins["kinetic_parameters"]["path"])
        return [f"    parameters = load_yaml('{path}')",""]

    # ================ INPUTS ============
    def _build_inputs(self):
        code = []
        code.append("    inputs = {}")
        code.append("")
        inputs = (self.model_ins["inputs"]["consider"])
        for var_name, cfg in inputs.items():
            path = cfg["path"]
            code.append(f"    {var_name}_func = import_function('{path}')")
            code.append(f"    {var_name} = {var_name}_func(t, self.br_id, '{var_name}')")
            code.append(f"    inputs['{var_name}'] = {var_name}")
            code.append("")
        return code

    # ================= VOLUME ======================
    def _build_volume_balance(self,volume):
        code = []
        path = (volume["feeds"]["flowrate"]["path"])
        code.append(f"    feed_profile = import_function('{path}')")
        feed_species = (volume["feeds"]["species"][0])
        code.append(f"    dVdt, I_F = feed_profile(t, self.br_id, '{feed_species}')")
        code.append("")
        return code

    # =============== KINETIC FUNCTIONS ===================
    def _build_kinetic_functions(self):
        code = []
        kinetic_functions = (self.model_ins["kinetic_functions"])
        code.append("    kinetics = {}")
        code.append("")
        for name, cfg in kinetic_functions.items():
            path = cfg["path"]
            code.append(f"    {name}_func = import_function('{path}')")
            code.append(f"    {name} = {name}_func(features, parameters)")
            code.append(f"    kinetics['{name}'] = {name}")
            code.append("")
        code.append("    features['kinetics'] = kinetics")
        code.append("")
        return code

    # ================ GENERATION TERMS =========================
    def _build_generation_term(self,state_cfg):
        generation = state_cfg["generation"]
        source = state_cfg["source"]
        code = []
        if source == "function":
            path = state_cfg["path"]
            code.append(f"    {generation}_func = import_function('{path}')")
            code.append(f"    {generation} = {generation}_func(features)")
        elif source == "constant":
            value = state_cfg["value"]
            code.append(f"    {generation} = {value}")
        elif source == "machine_learning":
            code.append(f"    # TODO ML model for {generation}")
        return code

    # ============= BALANCES =========================
    def _build_state_balance(self,state_name,state_cfg,volume):
        code = []
        code.extend(self._build_generation_term(state_cfg))
        generation = (state_cfg["generation"])
        V = volume["name"]
        equation = generation
        equation += (f" - ({state_name}/{V})*d{V}dt")
        fed_species = (volume["feeds"]["species"])
        if state_name in fed_species:
            concentration = (volume["feeds"]["concentration"][state_name])
            equation += (f" + ({concentration}/{V})*d{V}dt")
        code.append(f"    d{state_name}dt = {equation}")
        code.append("")

        return code
    
    def _build_features(self):
        code = []
        code.append("    features = {}")
        code.append("")
        code.append("    features.update(states)")
        code.append("    features.update(inputs)")
        code.append("")
        code.append("    features['states'] = states")
        code.append("    features['inputs'] = inputs")
        code.append("")
        return code