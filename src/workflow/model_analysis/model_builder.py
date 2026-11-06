
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
        code.append("")
        code.append("import numpy as np")
        code.append("from src.utils.io import load_yaml")
        code.extend(self._build_imports())
        code.append("")

        # --------- Function definition ------------
        code.append("def ODEs(self, t, state):")
        code.append("")

        # ------ States unpacking ----------
        code.append("    # --------- States --------")
        state_names = list(states.keys())
        code.append(f"    {', '.join(state_names + [volume['name']])} = state")
        code.append("    states = {")
        for s in state_names:
            code.append(f"        '{s}': {s},")
        code.append(f"        '{volume['name']}': {volume['name']}")
        code.append("    }")
        code.append("")
        
        # --------- Parameters --------
        code.append("    # --------- Parameters --------")
        code.extend(self._build_parameters())

        # -------------- Inputs ------------
        code.append("    # --------- Inputs --------")
        code.extend(self._build_inputs())

        # -------------- Features ------------
        code.append("    # --------- Features --------")
        code.extend(self._build_features())

        # --------- Volume Balance ----------
        code.append("    # --------- Volume --------")
        code.extend(self._build_volume_balance(volume))

        # ----------- Kinetics ------------------------
        code.append("    # --------- Kinetics --------")
        code.extend(self._build_kinetic_functions())

        # -------- State balances -----------------
        code.append("    # --------- Balances --------")
        for state_name, state_cfg in states.items():
            code.extend(self._build_state_balance(state_name,state_cfg,volume))

        # --------- Return ----------------------
        derivatives = [f"d{s}dt" for s in state_names]
        derivatives.append(f"d{volume['name']}dt")

        code.append(f"    return np.array([{', '.join(derivatives)}])")

        return "\n".join(code)
    
    def _build_imports(self):
        imports = {}

        # Inputs
        for _, cfg in self.model_ins["inputs"].items():
            module = cfg["path"]
            imports.setdefault(module,set()).add(cfg["function"])

        # Feed profile
        volume = self.model_ins["volume"]
        module = volume["feeds"]["flowrate"]["path"]
        function = (volume["feeds"]["flowrate"]["function"])
        imports.setdefault(module, set()).add(function)

        # Feed conceontration profile
        species = volume["feeds"]["species"][0]
        module = volume["feeds"]["concentration"][species]["path"]
        function = (volume["feeds"]["concentration"][species]["function"])
        imports.setdefault(module, set()).add(function)

        # Kinetics
        for _, cfg in self.model_ins["kinetic_functions"].items():
            module = cfg["path"]
            imports.setdefault(module,set()).add(cfg["function"])
        
        # Generation terms
        for _, cfg in self.model_ins["states"].items():
            if cfg["source"] != "function":
                continue
            module = cfg["path"]
            imports.setdefault(module, set()).add(cfg["function"])

        lines = []
        for module in sorted(imports):
            functions = sorted(imports[module])
            fn_string = ", ".join(sorted(functions))
            lines.append(f"from {module} import {fn_string}")
        lines.append("")
        return lines

    # =============== PARAMETERS ====================
    def _build_parameters(self):
        path = (self.model_ins["kinetic_parameters"]["path"])
        return [f"    parameters = load_yaml('{path}')",""]

    # ================ INPUTS ============
    def _build_inputs(self):
        code = []
        code.append("    inputs = {}")
        code.append("")
        inputs = self.model_ins["inputs"]
        for var_name, cfg in inputs.items():
            function = cfg["function"]
            code.append(f"    {var_name} = {function}(t, self.br_id, '{var_name}')")
            code.append(f"    inputs['{var_name}'] = {var_name}")
            code.append("")
        return code

    # ================= VOLUME ======================
    def _build_volume_balance(self, volume):
        code = []
        # ----- volume -----
        function = (volume["feeds"]["flowrate"]["function"])
        species = (volume["feeds"]["species"][0])
        code.append(f"    dVdt = {function}(t, self.br_id, '{species}')")
        # ----- titer -----
        titer = (volume["feeds"]["concentration"][species]["name"])
        titer_function = (volume["feeds"]["concentration"][species]["function"])
        code.append(f"    {titer} = {titer_function}(t, self.br_id, '{species}')")
        code.append("")
        return code

    # =============== KINETIC FUNCTIONS ===================
    def _build_kinetic_functions(self):
        code = []
        kinetics = self.model_ins["kinetic_functions"]
        code.append("    kinetics = {}")
        code.append("")
        for name, cfg in kinetics.items():
            if cfg["source"] == "function":
                function = cfg["function"]
                code.append(f"    {name} = {function}(features, parameters)")
                code.append(f"    kinetics['{name}'] = {name}")
                code.append("")
        code.append("    features['kinetics'] = kinetics")
        code.append("")
        return code

    # ================ GENERATION TERMS =========================
    def _build_generation_term(self, state_cfg):
        generation = state_cfg["generation"]
        source = state_cfg["source"]
        code = []
        if source == "function":
            function = state_cfg["function"]
            code.append(f"    {generation} = {function}(features, parameters)")
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
            concentration = (volume["feeds"]["concentration"][state_name]["name"])
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