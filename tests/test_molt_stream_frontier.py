from molt_stream.measurement.frontier import QualityEnergyPoint, pareto_frontier


def test_quality_energy_frontier_rejects_dominated_run():
    points = [
        QualityEnergyPoint("best", 10, 100, 5),
        QualityEnergyPoint("dominated", 11, 110, 6),
        QualityEnergyPoint("quality", 12, 120, 4),
    ]
    assert [point.run_id for point in pareto_frontier(points)] == ["best", "quality"]
