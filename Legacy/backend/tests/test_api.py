from fastapi.testclient import TestClient
from mars.api import app

client = TestClient(app)


def test_list_cases_returns_all_fixtures():
    response = client.get("/cases")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 15
    assert {"id", "label"} <= set(body[0].keys())


def test_get_volume_returns_slices_matching_shape():
    response = client.get("/cases/case_001/volume")
    assert response.status_code == 200
    body = response.json()
    assert body["shape"] == [16, 32, 32]
    assert len(body["slices"]) == 16


def test_get_volume_unknown_case_returns_404():
    response = client.get("/cases/not_a_real_case/volume")
    assert response.status_code == 404


def test_run_case_default_tau_returns_case_result():
    response = client.get("/cases/case_001/run")
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "case_001"
    assert body["tau"] == 0.5
    assert body["decision"] in ("auto", "escalate")


def test_run_case_custom_tau_changes_decision_for_borderline_case():
    low_tau = client.get("/cases/case_009/run", params={"tau": 0.01}).json()
    high_tau = client.get("/cases/case_009/run", params={"tau": 0.99}).json()
    assert low_tau["decision"] == "escalate"
    assert high_tau["decision"] == "auto"
