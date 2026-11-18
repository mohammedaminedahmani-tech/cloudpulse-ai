from src.common.schemas import MetricEvent


def test_aller_retour_json():
    e = MetricEvent(timestamp=300, cpu_demand=1.0, mem_demand=2.0, n_servers=30,
                    cpu_util=0.5, mem_util=0.6, saturated=False, missing_servers=0.0,
                    action="+1", q_values=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert MetricEvent.from_json(e.to_json()) == e