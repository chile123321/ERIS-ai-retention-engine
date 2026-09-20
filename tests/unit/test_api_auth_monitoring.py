"""Pure auth/tracing/metrics checks with no model or employee data."""

from uuid import UUID, uuid4

from eris_ml.api.auth import service_token_configured
from eris_ml.api.main import request_id_or_new
from eris_ml.api.monitoring import OperationalMetrics, safe_endpoint


def test_service_token_has_no_insecure_default_or_placeholder() -> None:
    assert not service_token_configured(None)
    assert not service_token_configured("")
    assert not service_token_configured("short")
    assert not service_token_configured("<SET_A_LONG_RANDOM_TOKEN_IN_RUNTIME_ENV>")
    assert not service_token_configured(" " + "x" * 40)
    assert service_token_configured("synthetic-test-only-service-credential-0001")


def test_request_id_is_canonical_uuid_only() -> None:
    valid = str(uuid4())
    assert request_id_or_new(valid) == valid
    for invalid in (None, "EMPLOYEE-42", "x" * 100, valid.upper()):
        replacement = request_id_or_new(invalid)
        assert replacement != invalid
        assert str(UUID(replacement)) == replacement


def test_metrics_labels_are_static_and_totals_are_consistent() -> None:
    metrics = OperationalMetrics()
    metrics.set_ready(True)
    metrics.record_request("/employee/PRIVATE_42", "GET", 404, 0.1)
    metrics.record_request("/api/v1/predict", "POST", 422, 0.2)
    metrics.record_request("/api/v1/model-info", "GET", 401, 0.3)
    metrics.increment("unknown_category_warnings")
    metrics.increment("prediction_alerts", 2)
    rendered = metrics.render()
    assert safe_endpoint("/employee/PRIVATE_42") == "unmatched"
    assert "PRIVATE_42" not in rendered
    assert 'endpoint="unmatched",method="GET",status="404"} 1' in rendered
    assert "eris_prediction_requests_total 1" in rendered
    assert "eris_validation_failures_total 1" in rendered
    assert "eris_authentication_failures_total 1" in rendered
    assert "eris_prediction_alerts_total 2" in rendered
    assert "eris_unknown_category_warnings_total 1" in rendered
    assert "eris_api_ready 1" in rendered
