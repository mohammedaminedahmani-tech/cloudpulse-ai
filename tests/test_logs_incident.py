import json

import numpy as np

from src.logs.calibration import ece, fit_temperature, softmax
from src.logs.incident import IncidentEvent


def test_incident_aller_retour_json():
    e = IncidentEvent(timestamp=260700, log_text="RAS KERNEL FATAL program interrupt",
                      label="anomalie", anomaly_score=0.93, confidence=0.93, model="lora+ewc")
    e2 = IncidentEvent.from_json(e.to_json())
    assert e2 == e
    d = json.loads(e.to_json())
    assert d["source"] == "cloudpulse.logs.classifier" and d["schema_version"] == "1.0"


def test_ece_nulle_si_parfait():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert ece(probs, np.array([0, 1, 0])) == 0.0


def test_temperature_corrige_un_modele_trop_confiant():
    rng = np.random.RandomState(0)
    z = rng.normal(size=(3000, 2)) * 1.5
    p = softmax(z)
    y = np.array([rng.choice(2, p=pi) for pi in p])      # étiquettes cohérentes avec z
    T = fit_temperature(3.0 * z, y)                       # modèle 3 fois trop confiant
    assert 1.5 < T < 5.0
    assert ece(softmax(3.0 * z, T), y) < ece(softmax(3.0 * z), y)