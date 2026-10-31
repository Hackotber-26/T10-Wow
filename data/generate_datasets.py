"""
Generates rich, realistic cloud observability datasets for RootSight.
Covers 6 distinct scenarios:
  1. database_latency_spike (postgres-db failure)
  2. payment_service_failure (payment-service crash/500s)
  3. api_latency_spike (api-gateway regex stall)
  4. inventory_service_overload (inventory-service CPU/RAM saturation)
  5. normal_healthy_system (healthy baseline)
  6. insufficient_evidence (sparse observations, no clear anomaly)
"""

import os
import json
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_all_data():
    incidents = [
        {
            "incident_id": "INC-001-DB-SPIKE",
            "name": "Database Latency Spike",
            "description": "Lock contention on relation order_items in postgres-db causing cascaded cart and checkout timeouts.",
            "ground_truth_root_cause": "postgres-db",
            "category": "database_lock",
            "services": ["frontend", "api-gateway", "cart-service", "checkout-service", "postgres-db", "redis-cache"],
        },
        {
            "incident_id": "INC-002-PAY-FAIL",
            "name": "Payment Service Failure",
            "description": "Payment service throws 500 internal errors and 85% error rate due to expired gateway SSL certificate.",
            "ground_truth_root_cause": "payment-service",
            "category": "service_crash",
            "services": ["frontend", "api-gateway", "checkout-service", "payment-service", "notification-service"],
        },
        {
            "incident_id": "INC-003-API-GATEWAY",
            "name": "API Gateway Latency Spike",
            "description": "ReDoS vulnerability / route matching thread stall in API gateway causing frontend latency surge.",
            "ground_truth_root_cause": "api-gateway",
            "category": "gateway_bottleneck",
            "services": ["frontend", "api-gateway", "user-service", "product-service"],
        },
        {
            "incident_id": "INC-004-INV-OVERLOAD",
            "name": "Inventory Service Overload",
            "description": "Flash sale traffic saturates inventory-service CPU to 98% and memory to 94% with queued requests.",
            "ground_truth_root_cause": "inventory-service",
            "category": "resource_exhaustion",
            "services": ["frontend", "api-gateway", "product-service", "inventory-service", "warehouse-db"],
        },
        {
            "incident_id": "INC-005-HEALTHY",
            "name": "Normal Healthy System",
            "description": "Normal steady-state traffic across all microservices with no anomalous deviations.",
            "ground_truth_root_cause": "None",
            "category": "healthy",
            "services": ["frontend", "api-gateway", "user-service", "cart-service", "postgres-db"],
        },
        {
            "incident_id": "INC-006-INSUFFICIENT",
            "name": "Insufficient Telemetry Evidence",
            "description": "Sparse metrics and missing trace logs prevent conclusive deterministic diagnosis.",
            "ground_truth_root_cause": "unknown",
            "category": "inconclusive",
            "services": ["frontend", "analytics-worker"],
        },
    ]

    metrics_rows = []
    logs_rows = []
    traces_rows = []
    deps_rows = []

    # 1. INC-001: Database Latency Spike
    # Baseline 10 timesteps, spike at timesteps 11-15
    for t in range(1, 16):
        ts = f"2026-10-30T14:{t:02d}:00Z"
        is_spike = (t >= 11)

        # postgres-db (Root cause: latency surges from 15ms to 980ms)
        pg_lat = np.random.normal(950, 40) if is_spike else np.random.normal(15, 2)
        pg_cpu = np.random.normal(88, 3) if is_spike else np.random.normal(35, 3)
        metrics_rows.extend([
            {"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "postgres-db", "metric_name": "latency_ms", "metric_value": round(float(pg_lat), 2)},
            {"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "postgres-db", "metric_name": "cpu_percent", "metric_value": round(float(pg_cpu), 2)},
            {"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "postgres-db", "metric_name": "error_rate", "metric_value": 0.08 if is_spike else 0.0},
        ])

        # cart-service (Calls postgres: latency surges from 30ms to 990ms)
        cart_lat = np.random.normal(1020, 50) if is_spike else np.random.normal(30, 4)
        metrics_rows.extend([
            {"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "cart-service", "metric_name": "latency_ms", "metric_value": round(float(cart_lat), 2)},
            {"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "cart-service", "metric_name": "error_rate", "metric_value": 0.05 if is_spike else 0.0},
        ])

        # api-gateway (Calls cart: latency increases)
        gw_lat = np.random.normal(1100, 60) if is_spike else np.random.normal(45, 5)
        metrics_rows.append({"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "api-gateway", "metric_name": "latency_ms", "metric_value": round(float(gw_lat), 2)})

        # redis-cache (Healthy dependency)
        metrics_rows.append({"incident_id": "INC-001-DB-SPIKE", "timestamp": ts, "service": "redis-cache", "metric_name": "latency_ms", "metric_value": round(float(np.random.normal(3, 0.5)), 2)})

    # Logs for INC-001
    logs_rows.extend([
        {"incident_id": "INC-001-DB-SPIKE", "timestamp": "2026-10-30T14:11:15Z", "service": "postgres-db", "level": "ERROR", "message": "ERROR: query wait lock timeout on relation order_items after 30000ms", "log_id": "LOG-PG-001"},
        {"incident_id": "INC-001-DB-SPIKE", "timestamp": "2026-10-30T14:11:45Z", "service": "postgres-db", "level": "ERROR", "message": "FATAL: deadlock detected while waiting for ShareLock on transaction 849201", "log_id": "LOG-PG-002"},
        {"incident_id": "INC-001-DB-SPIKE", "timestamp": "2026-10-30T14:12:10Z", "service": "cart-service", "level": "ERROR", "message": "HikariPool-1 - Connection is not available, request timed out after 30000ms", "log_id": "LOG-CART-001"},
        {"incident_id": "INC-001-DB-SPIKE", "timestamp": "2026-10-30T14:12:30Z", "service": "api-gateway", "level": "WARN", "message": "Upstream service 'cart-service' slow response: 1050ms", "log_id": "LOG-GW-001"},
        {"incident_id": "INC-001-DB-SPIKE", "timestamp": "2026-10-30T14:05:00Z", "service": "postgres-db", "level": "INFO", "message": "Checkpoint complete: wrote 842 buffers (5.1%)", "log_id": "LOG-PG-003"},
    ])

    # Traces for INC-001
    traces_rows.extend([
        {"incident_id": "INC-001-DB-SPIKE", "trace_id": "TR-001-A", "span_id": "SP-01", "parent_span_id": None, "source_service": "frontend", "destination_service": "api-gateway", "latency_ms": 1150.0, "timestamp": "2026-10-30T14:12:00Z"},
        {"incident_id": "INC-001-DB-SPIKE", "trace_id": "TR-001-A", "span_id": "SP-02", "parent_span_id": "SP-01", "source_service": "api-gateway", "destination_service": "cart-service", "latency_ms": 1080.0, "timestamp": "2026-10-30T14:12:00Z"},
        {"incident_id": "INC-001-DB-SPIKE", "trace_id": "TR-001-A", "span_id": "SP-03", "parent_span_id": "SP-02", "source_service": "cart-service", "destination_service": "postgres-db", "latency_ms": 995.0, "timestamp": "2026-10-30T14:12:01Z"},
        {"incident_id": "INC-001-DB-SPIKE", "trace_id": "TR-001-B", "span_id": "SP-04", "parent_span_id": None, "source_service": "cart-service", "destination_service": "redis-cache", "latency_ms": 4.5, "timestamp": "2026-10-30T14:12:02Z"},
    ])

    # Dependencies for INC-001
    deps_rows.extend([
        {"incident_id": "INC-001-DB-SPIKE", "source_service": "frontend", "destination_service": "api-gateway", "call_count": 500},
        {"incident_id": "INC-001-DB-SPIKE", "source_service": "api-gateway", "destination_service": "cart-service", "call_count": 350},
        {"incident_id": "INC-001-DB-SPIKE", "source_service": "cart-service", "destination_service": "postgres-db", "call_count": 350},
        {"incident_id": "INC-001-DB-SPIKE", "source_service": "cart-service", "destination_service": "redis-cache", "call_count": 200},
        {"incident_id": "INC-001-DB-SPIKE", "source_service": "checkout-service", "destination_service": "postgres-db", "call_count": 120},
    ])

    # 2. INC-002: Payment Service Failure
    for t in range(1, 16):
        ts = f"2026-10-30T15:{t:02d}:00Z"
        is_spike = (t >= 10)

        pay_err = np.random.uniform(0.75, 0.92) if is_spike else 0.001
        pay_lat = np.random.normal(850, 60) if is_spike else np.random.normal(120, 10)
        metrics_rows.extend([
            {"incident_id": "INC-002-PAY-FAIL", "timestamp": ts, "service": "payment-service", "metric_name": "error_rate", "metric_value": round(float(pay_err), 4)},
            {"incident_id": "INC-002-PAY-FAIL", "timestamp": ts, "service": "payment-service", "metric_name": "latency_ms", "metric_value": round(float(pay_lat), 2)},
        ])

        chk_err = np.random.uniform(0.35, 0.50) if is_spike else 0.002
        metrics_rows.append({"incident_id": "INC-002-PAY-FAIL", "timestamp": ts, "service": "checkout-service", "metric_name": "error_rate", "metric_value": round(float(chk_err), 4)})
        metrics_rows.append({"incident_id": "INC-002-PAY-FAIL", "timestamp": ts, "service": "notification-service", "metric_name": "error_rate", "metric_value": 0.0})

    logs_rows.extend([
        {"incident_id": "INC-002-PAY-FAIL", "timestamp": "2026-10-30T15:10:05Z", "service": "payment-service", "level": "ERROR", "message": "SSLHandshakeException: PKIX path validation failed: certificate expired 2026-10-30T00:00:00Z", "log_id": "LOG-PAY-001"},
        {"incident_id": "INC-002-PAY-FAIL", "timestamp": "2026-10-30T15:10:20Z", "service": "payment-service", "level": "FATAL", "message": "PaymentGatewayClientException: 503 Service Unavailable on POST /v2/charges", "log_id": "LOG-PAY-002"},
        {"incident_id": "INC-002-PAY-FAIL", "timestamp": "2026-10-30T15:11:00Z", "service": "checkout-service", "level": "ERROR", "message": "Call to payment-service failed with HTTP 500 Internal Server Error", "log_id": "LOG-CHK-001"},
    ])

    traces_rows.extend([
        {"incident_id": "INC-002-PAY-FAIL", "trace_id": "TR-002-A", "span_id": "SP-20", "parent_span_id": None, "source_service": "api-gateway", "destination_service": "checkout-service", "latency_ms": 900.0, "timestamp": "2026-10-30T15:11:00Z"},
        {"incident_id": "INC-002-PAY-FAIL", "trace_id": "TR-002-A", "span_id": "SP-21", "parent_span_id": "SP-20", "source_service": "checkout-service", "destination_service": "payment-service", "latency_ms": 850.0, "timestamp": "2026-10-30T15:11:00Z"},
        {"incident_id": "INC-002-PAY-FAIL", "trace_id": "TR-002-B", "span_id": "SP-22", "parent_span_id": None, "source_service": "checkout-service", "destination_service": "notification-service", "latency_ms": 15.0, "timestamp": "2026-10-30T15:11:02Z"},
    ])

    deps_rows.extend([
        {"incident_id": "INC-002-PAY-FAIL", "source_service": "api-gateway", "destination_service": "checkout-service", "call_count": 400},
        {"incident_id": "INC-002-PAY-FAIL", "source_service": "checkout-service", "destination_service": "payment-service", "call_count": 400},
        {"incident_id": "INC-002-PAY-FAIL", "source_service": "checkout-service", "destination_service": "notification-service", "call_count": 150},
    ])

    # 3. INC-003: API Gateway Latency Spike
    for t in range(1, 16):
        ts = f"2026-10-30T16:{t:02d}:00Z"
        is_spike = (t >= 8)

        gw_lat = np.random.normal(1450, 70) if is_spike else np.random.normal(25, 3)
        gw_cpu = np.random.normal(96, 2) if is_spike else np.random.normal(30, 4)
        metrics_rows.extend([
            {"incident_id": "INC-003-API-GATEWAY", "timestamp": ts, "service": "api-gateway", "metric_name": "latency_ms", "metric_value": round(float(gw_lat), 2)},
            {"incident_id": "INC-003-API-GATEWAY", "timestamp": ts, "service": "api-gateway", "metric_name": "cpu_percent", "metric_value": round(float(gw_cpu), 2)},
            # Backends remain completely normal!
            {"incident_id": "INC-003-API-GATEWAY", "timestamp": ts, "service": "user-service", "metric_name": "latency_ms", "metric_value": round(float(np.random.normal(12, 2)), 2)},
            {"incident_id": "INC-003-API-GATEWAY", "timestamp": ts, "service": "product-service", "metric_name": "latency_ms", "metric_value": round(float(np.random.normal(18, 2)), 2)},
        ])

    logs_rows.extend([
        {"incident_id": "INC-003-API-GATEWAY", "timestamp": "2026-10-30T16:08:12Z", "service": "api-gateway", "level": "ERROR", "message": "Regular expression evaluation timeout in path matcher: [a-z0-9._%+-]+@...", "log_id": "LOG-GW-010"},
        {"incident_id": "INC-003-API-GATEWAY", "timestamp": "2026-10-30T16:09:00Z", "service": "api-gateway", "level": "ERROR", "message": "Worker thread pool starvation: 64/64 threads busy in regex matcher", "log_id": "LOG-GW-011"},
    ])

    traces_rows.extend([
        {"incident_id": "INC-003-API-GATEWAY", "trace_id": "TR-003-A", "span_id": "SP-30", "parent_span_id": None, "source_service": "frontend", "destination_service": "api-gateway", "latency_ms": 1500.0, "timestamp": "2026-10-30T16:09:00Z"},
        {"incident_id": "INC-003-API-GATEWAY", "trace_id": "TR-003-A", "span_id": "SP-31", "parent_span_id": "SP-30", "source_service": "api-gateway", "destination_service": "user-service", "latency_ms": 12.0, "timestamp": "2026-10-30T16:09:01Z"},
    ])

    deps_rows.extend([
        {"incident_id": "INC-003-API-GATEWAY", "source_service": "frontend", "destination_service": "api-gateway", "call_count": 800},
        {"incident_id": "INC-003-API-GATEWAY", "source_service": "api-gateway", "destination_service": "user-service", "call_count": 400},
        {"incident_id": "INC-003-API-GATEWAY", "source_service": "api-gateway", "destination_service": "product-service", "call_count": 400},
    ])

    # 4. INC-004: Inventory Service Overload
    for t in range(1, 16):
        ts = f"2026-10-30T17:{t:02d}:00Z"
        is_spike = (t >= 9)

        inv_cpu = np.random.normal(97.5, 1.2) if is_spike else np.random.normal(40, 5)
        inv_mem = np.random.normal(94.2, 1.5) if is_spike else np.random.normal(55, 3)
        inv_lat = np.random.normal(820, 45) if is_spike else np.random.normal(45, 4)

        metrics_rows.extend([
            {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": ts, "service": "inventory-service", "metric_name": "cpu_percent", "metric_value": round(float(inv_cpu), 2)},
            {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": ts, "service": "inventory-service", "metric_name": "memory_percent", "metric_value": round(float(inv_mem), 2)},
            {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": ts, "service": "inventory-service", "metric_name": "latency_ms", "metric_value": round(float(inv_lat), 2)},
            {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": ts, "service": "product-service", "metric_name": "latency_ms", "metric_value": round(float(inv_lat + 20), 2) if is_spike else 55.0},
            {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": ts, "service": "warehouse-db", "metric_name": "cpu_percent", "metric_value": 32.0},
        ])

    logs_rows.extend([
        {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": "2026-10-30T17:09:30Z", "service": "inventory-service", "level": "ERROR", "message": "OutOfMemoryError: Java heap space during batch inventory reservation", "log_id": "LOG-INV-001"},
        {"incident_id": "INC-004-INV-OVERLOAD", "timestamp": "2026-10-30T17:10:00Z", "service": "inventory-service", "level": "ERROR", "message": "High CPU throttle: cgroup throttled 420 periods in last 10s", "log_id": "LOG-INV-002"},
    ])

    traces_rows.extend([
        {"incident_id": "INC-004-INV-OVERLOAD", "trace_id": "TR-004-A", "span_id": "SP-40", "parent_span_id": None, "source_service": "api-gateway", "destination_service": "product-service", "latency_ms": 850.0, "timestamp": "2026-10-30T17:10:00Z"},
        {"incident_id": "INC-004-INV-OVERLOAD", "trace_id": "TR-004-A", "span_id": "SP-41", "parent_span_id": "SP-40", "source_service": "product-service", "destination_service": "inventory-service", "latency_ms": 810.0, "timestamp": "2026-10-30T17:10:00Z"},
        {"incident_id": "INC-004-INV-OVERLOAD", "trace_id": "TR-004-A", "span_id": "SP-42", "parent_span_id": "SP-41", "source_service": "inventory-service", "destination_service": "warehouse-db", "latency_ms": 14.0, "timestamp": "2026-10-30T17:10:01Z"},
    ])

    deps_rows.extend([
        {"incident_id": "INC-004-INV-OVERLOAD", "source_service": "api-gateway", "destination_service": "product-service", "call_count": 600},
        {"incident_id": "INC-004-INV-OVERLOAD", "source_service": "product-service", "destination_service": "inventory-service", "call_count": 600},
        {"incident_id": "INC-004-INV-OVERLOAD", "source_service": "inventory-service", "destination_service": "warehouse-db", "call_count": 300},
    ])

    # 5. INC-005: Healthy Steady State
    for t in range(1, 11):
        ts = f"2026-10-30T18:{t:02d}:00Z"
        for s, base_lat, base_cpu in [("frontend", 15, 25), ("api-gateway", 20, 30), ("user-service", 22, 35), ("cart-service", 28, 40), ("postgres-db", 14, 28)]:
            metrics_rows.extend([
                {"incident_id": "INC-005-HEALTHY", "timestamp": ts, "service": s, "metric_name": "latency_ms", "metric_value": round(float(np.random.normal(base_lat, 1.5)), 2)},
                {"incident_id": "INC-005-HEALTHY", "timestamp": ts, "service": s, "metric_name": "cpu_percent", "metric_value": round(float(np.random.normal(base_cpu, 2)), 2)},
                {"incident_id": "INC-005-HEALTHY", "timestamp": ts, "service": s, "metric_name": "error_rate", "metric_value": 0.0},
            ])

    logs_rows.extend([
        {"incident_id": "INC-005-HEALTHY", "timestamp": "2026-10-30T18:05:00Z", "service": "api-gateway", "level": "INFO", "message": "Health check OK 200", "log_id": "LOG-HLT-001"},
        {"incident_id": "INC-005-HEALTHY", "timestamp": "2026-10-30T18:05:10Z", "service": "postgres-db", "level": "INFO", "message": "Auto-vacuum completed on pg_catalog.pg_class", "log_id": "LOG-HLT-002"},
    ])

    traces_rows.extend([
        {"incident_id": "INC-005-HEALTHY", "trace_id": "TR-005-A", "span_id": "SP-50", "parent_span_id": None, "source_service": "api-gateway", "destination_service": "user-service", "latency_ms": 21.0, "timestamp": "2026-10-30T18:05:00Z"},
        {"incident_id": "INC-005-HEALTHY", "trace_id": "TR-005-B", "span_id": "SP-51", "parent_span_id": None, "source_service": "cart-service", "destination_service": "postgres-db", "latency_ms": 13.5, "timestamp": "2026-10-30T18:05:01Z"},
    ])

    deps_rows.extend([
        {"incident_id": "INC-005-HEALTHY", "source_service": "api-gateway", "destination_service": "user-service", "call_count": 500},
        {"incident_id": "INC-005-HEALTHY", "source_service": "api-gateway", "destination_service": "cart-service", "call_count": 400},
        {"incident_id": "INC-005-HEALTHY", "source_service": "cart-service", "destination_service": "postgres-db", "call_count": 400},
    ])

    # 6. INC-006: Insufficient Telemetry Evidence
    # Only 2 points, perfectly normal or ambiguous, no traces
    metrics_rows.extend([
        {"incident_id": "INC-006-INSUFFICIENT", "timestamp": "2026-10-30T19:00:00Z", "service": "frontend", "metric_name": "latency_ms", "metric_value": 35.0},
        {"incident_id": "INC-006-INSUFFICIENT", "timestamp": "2026-10-30T19:01:00Z", "service": "analytics-worker", "metric_name": "cpu_percent", "metric_value": 45.0},
    ])

    logs_rows.append(
        {"incident_id": "INC-006-INSUFFICIENT", "timestamp": "2026-10-30T19:00:10Z", "service": "analytics-worker", "level": "INFO", "message": "Batch cycle started", "log_id": "LOG-INS-001"}
    )

    # Save to disk
    df_metrics = pd.DataFrame(metrics_rows)
    df_logs = pd.DataFrame(logs_rows)
    df_traces = pd.DataFrame(traces_rows)
    df_deps = pd.DataFrame(deps_rows)

    out_dir = os.path.dirname(__file__)
    df_metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    df_logs.to_csv(os.path.join(out_dir, "logs.csv"), index=False)
    df_traces.to_csv(os.path.join(out_dir, "traces.csv"), index=False)
    df_deps.to_csv(os.path.join(out_dir, "dependencies.csv"), index=False)

    with open(os.path.join(out_dir, "incidents.json"), "w") as f:
        json.dump(incidents, f, indent=2)

    print(f"Generated {len(df_metrics)} metric records, {len(df_logs)} log records, {len(df_traces)} trace spans across 6 scenarios.")

if __name__ == "__main__":
    generate_all_data()
