"""Recoge tres mezclas gestuales; la aceptación visual sigue siendo humana."""


class MorphGuide:
    STAGES = (
        ("closed", "JUNTA pulgar e indice IZQUIERDOS y mantenlos: buscamos 0%"),
        ("middle", "ABRE la pinza IZQUIERDA a MEDIA distancia: mezcla intermedia"),
        ("open", "SEPARA al maximo calibrado pulgar e indice IZQUIERDOS: intensidad 100%"),
    )

    def __init__(self):
        self.index = 0
        self.since = None
        self.samples = 0
        self.results = []

    @property
    def done(self):
        return self.index == len(self.STAGES)

    def status(self):
        if self.done:
            return ("TRES MEZCLAS CAPTURADAS. Revisa el resultado y cierra con Q.",
                    "La calidad visual se confirma despues; M permite ver triangulos.")
        return (self.STAGES[self.index][1],
                "Mira de frente, sin tapar la cara. Se guarda al mantener 2 s estables.")

    def update(self, result, gesture, now, fixed=False):
        if self.done:
            return None
        alpha = result.control_alpha
        expected = (alpha == 0, .3 <= alpha <= .7, alpha == 1)[self.index]
        valid = (not fixed and gesture["state"] == "tracking" and expected
                 and result.status == ("webcam" if self.index == 0 else "morphing"))
        if not valid:
            self.since, self.samples = None, 0
            return None
        if self.since is None:
            self.since = now
        self.samples += 1
        if now - self.since < 2 or self.samples < 8:
            return None
        name = self.STAGES[self.index][0]
        self.results.append({"stage": name, "alpha": result.alpha, "control_alpha": alpha, "samples": self.samples,
                             "hold_seconds": now - self.since, "time": now})
        self.index += 1
        self.since, self.samples = None, 0
        return name

    def summary(self):
        return {"completed": self.done, "stage": "done" if self.done else self.STAGES[self.index][0],
                "results": self.results, "visual_acceptance": "pending"}
