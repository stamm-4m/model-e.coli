
import numpy as np
from scipy.integrate import solve_ivp
from types import SimpleNamespace

class Simulator:
    def __init__(self, model, method, rtol, atol):
        self.model = model
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def run(self, y0, t_span, n_points=None, t_eval=None):   
        
        if t_eval is None:
            if n_points is None:
                raise ValueError("Provide either n_points or t_eval")
            t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # t0, tf = t_span

        # # Obtener discontinuidades del modelo
        # if hasattr(self.model, "get_discontinuity_times"):
        #     breaks = self.model.get_discontinuity_times(t0, tf)
        # else:
        #     breaks = []

        # # Añadir el final
        # segments = breaks + [tf]

        # t_all = []
        # y_all = []

        # y_init = np.array(y0, dtype=float)
        # t_start = t0

        # for t_end in segments:

        #     # t_eval del segmento
        #     mask = (t_eval >= t_start) & (t_eval <= t_end)
        #     t_eval_seg = t_eval[mask]

        #     # Nada que integrar en este tramo
        #     if len(t_eval_seg) == 0:
        #         t_start = t_end
        #         continue

        #     sol = solve_ivp(
        #         fun=self.model.dfdt,
        #         t_span=(t_start, t_end),
        #         y0=y_init,
        #         t_eval=t_eval_seg,
        #         method=self.method,
        #         rtol=self.rtol,
        #         atol=self.atol
        #     )

        #     if not sol.success:
        #         raise RuntimeError(sol.message)

        #     # Evitar duplicar el punto inicial
        #     if len(t_all) > 0:
        #         t_all.append(sol.t[1:])
        #         y_all.append(sol.y[:, 1:])
        #     else:
        #         t_all.append(sol.t)
        #         y_all.append(sol.y)

        #     # Preparar siguiente tramo
        #     y_init = sol.y[:, -1]
        #     t_start = t_end

        # # Reconstruir solución "tipo solve_ivp"
        # sol = SimpleNamespace(
        #     t=np.concatenate(t_all),
        #     y=np.concatenate(y_all, axis=1),
        #     success=True
        # )


        sol = solve_ivp(
            fun=self.model.dfdt,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )

        return sol
