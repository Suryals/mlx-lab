"""
bench/ep03_tools.py — Mock tool world for the Ep 03 agent eval.

Nine tools spanning dataops, cloud/k8s, frontend, backend, auth, and linux.
All outputs are deterministic canned data so runs are exactly reproducible
and scoring never depends on a live system.

TOOL_SCHEMAS is OpenAI/OpenRouter function-calling format; the MLX engine
passes the same schemas into the chat template (Qwen understands this format
natively via its Hermes-style tool template).
"""

import json

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bq_job_status",
            "description": "Get the status and error details of a BigQuery job by its job ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The BigQuery job ID, e.g. 'bq_load_daily_sales'."},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cloud_observability",
            "description": "Query a metric time series for a backend service (CPU, memory, error rate, latency).",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name, e.g. 'payments-api'."},
                    "metric": {"type": "string", "enum": ["cpu", "memory", "error_rate", "latency"]},
                    "window": {"type": "string", "description": "Lookback window, e.g. '1h' or '24h'. Default '1h'."},
                },
                "required": ["service", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_pod_status",
            "description": "Get pod status, restart counts, and last termination reason for a Kubernetes deployment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "deployment": {"type": "string"},
                },
                "required": ["namespace", "deployment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_console_logs",
            "description": "Fetch recent browser console errors/warnings captured by RUM for a frontend page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_url": {"type": "string", "description": "Page path, e.g. '/dashboard'."},
                },
                "required": ["page_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lighthouse_audit",
            "description": "Run a Lighthouse audit against a page and return the score plus top findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_url": {"type": "string"},
                    "category": {"type": "string", "enum": ["performance", "accessibility", "seo"]},
                },
                "required": ["page_url", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_probe",
            "description": "Send a test request to an internal API endpoint and return status code, latency, and body excerpt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "Path, e.g. '/api/orders'."},
                    "method": {"type": "string", "enum": ["GET", "POST", "HEAD"]},
                },
                "required": ["endpoint", "method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "idp_user_lookup",
            "description": "Look up a user's account state in the identity provider (lock status, MFA enrollment, last login).",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "host_command",
            "description": "Run a read-only diagnostic on a Linux host. Commands: disk_usage (df), memory (free/vmstat), service_status (systemctl status of `service`), uptime (load averages).",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Hostname, e.g. 'web-01'."},
                    "command": {"type": "string", "enum": ["disk_usage", "memory", "service_status", "uptime"]},
                    "service": {"type": "string", "description": "Required when command is 'service_status'."},
                },
                "required": ["host", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ticket_lookup",
            "description": "Fetch an incident ticket by ID: summary, status, and resolution notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "e.g. 'INC-1234'."},
                },
                "required": ["ticket_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Canned world state
# ---------------------------------------------------------------------------

_BQ_JOBS = {
    "bq_load_daily_sales": {
        "state": "FAILED",
        "started": "2026-08-15T02:10:00Z",
        "ended": "2026-08-15T02:14:12Z",
        "error": "quotaExceeded: slot quota exhausted for reservation 'etl-slots'",
    },
    "bq_export_events": {"state": "SUCCEEDED", "started": "2026-08-15T01:00:00Z", "ended": "2026-08-15T01:07:41Z", "error": None},
    "bq_transform_orders": {"state": "RUNNING", "started": "2026-08-15T04:00:00Z", "ended": None, "error": None},
    # --- hard tier ---
    "sync_crm_to_bq_88": {
        "state": "FAILED",
        "started": "2026-08-16T01:00:00Z",
        "ended": "2026-08-16T01:42:10Z",
        "error": "readTimeout: upstream CRM API not responding in time; check latency of service 'crm-backend' over a 1h window",
    },
}

_METRICS = {
    ("payments-api", "cpu"): {"current": "82%", "p95": "88%", "trend": "elevated for 40m", "threshold": "80%"},
    ("payments-api", "error_rate"): {"current": "0.2%", "trend": "flat/stable", "threshold": "1%"},
    ("kafka-ingest", "error_rate"): {"current": "7.4%", "trend": "spike began 03:00 UTC", "threshold": "1%"},
    ("bigquery-loader", "error_rate"): {"current": "0.1%", "trend": "stable over window", "threshold": "1%"},
    ("dataproc-etl", "memory"): {"current": "94%", "trend": "climbing since job start", "threshold": "90%"},
    ("orders-db", "latency"): {"current": "4200ms p95", "trend": "connection pool saturated, queries timing out", "threshold": "500ms"},
    # --- hard tier ---
    ("crm-backend", "latency"): {"current": "9800ms p95", "trend": "degraded; host crm-node-01 is I/O-bound — disk nearly full, check disk usage on crm-node-01", "threshold": "800ms"},
    ("auth-service", "memory"): {"current": "31%", "trend": "flat — no growth before crashes; containers exit at startup, inconsistent with a memory leak", "threshold": "90%"},
}

_PODS = {
    ("prod", "checkout"): {
        "pods": 3,
        "ready": 1,
        "restarts_last_hour": 14,
        "last_state": "CrashLoopBackOff",
        "last_termination_reason": "OOMKilled (memory limit 512Mi exceeded)",
    },
    # --- hard tier ---
    ("core-infra", "auth-service"): {
        "pods": 3,
        "ready": 0,
        "restarts_last_hour": 22,
        "last_state": "CrashLoopBackOff",
        "last_termination_reason": "Error: InvalidConfiguration — mounted secret 'auth-jwt-keys' not found; container exits during startup (NOT OOMKilled)",
    },
}

_CONSOLE_LOGS = {
    "/dashboard": [
        {"level": "error", "msg": "Uncaught TypeError: Cannot read properties of undefined (reading 'widgets') at DashboardRoot.render (main.js:2411)"},
        {"level": "warning", "msg": "React hydration mismatch suppressed"},
    ],
    # --- hard tier ---
    "https://app.global/reports": [
        {"level": "error", "msg": "GET /api/reports 401 Unauthorized — backend data service authentication failed for principal eu-service-account@global.com"},
        {"level": "error", "msg": "ReportGrid.render: data undefined after fetch failure — rendering blank state"},
    ],
}

_LIGHTHOUSE = {
    ("/landing", "performance"): {
        "score": 58,
        "top_findings": [
            "Largest Contentful Paint 6.1s — hero image (4.2MB PNG) unoptimized, no srcset",
            "Render-blocking script analytics.js 380ms",
        ],
    },
    # --- hard tier ---
    ("https://app.global/reports", "performance"): {
        "score": 34,
        "top_findings": [
            "Main XHR to /api/reports blocks render for 28s, then fails",
            "Repeated 401 responses from /api/reports (retry storm) — see browser console for the failing principal",
        ],
    },
}

_API = {
    ("/api/orders", "GET"): {"status_code": 500, "latency_ms": 5012, "body": '{"error":"upstream database timeout after 5000ms"}'},
    ("/api/health", "GET"): {"status_code": 200, "latency_ms": 12, "body": '{"status":"ok"}'},
}

_IDP_USERS = {
    "priya@corp.com": {"status": "LOCKED", "reason": "5 failed password attempts at 06:02 UTC", "mfa_enrolled": True, "last_login": "2026-08-14T18:22:00Z"},
    "dev@corp.com": {"status": "ACTIVE", "reason": "MFA enrollment incomplete — conditional access policy blocks sign-in until MFA is enrolled", "mfa_enrolled": False, "last_login": "2026-08-10T09:12:00Z"},
    # --- hard tier ---
    "eu-service-account@global.com": {"status": "SUSPENDED", "reason": "Mandatory password rotation not completed by deadline — account suspended by security policy", "mfa_enrolled": True, "last_login": "2026-07-30T06:00:00Z"},
    "ceo@enterprise.com": {"status": "ACTIVE", "reason": "Sign-in from unenrolled mobile device blocked by conditional access policy 'Block-Untrusted-Mobile-Devices' (device must be MDM-enrolled)", "mfa_enrolled": True, "last_login": "2026-08-16T07:45:00Z"},
}

_HOSTS = {
    ("web-01", "disk_usage"): "Filesystem  Size  Used  Avail  Use%  Mounted on\n/dev/sda1   50G   21G   29G    42%  /\n/dev/sda2   100G  97G   3G     97%  /var",
    ("web-01", "memory"): "total 16G, used 15.2G, free 0.3G, swap total 8G, swap used 7.6G — heavy swapping, si/so high",
    ("web-01", "uptime"): "up 84 days, load average: 9.12, 8.77, 8.02 (4 cores)",
    ("web-02", "disk_usage"): "Filesystem  Size  Used  Avail  Use%  Mounted on\n/dev/sda1   50G   18G   32G    36%  /",
    (("web-02", "service_status"), "nginx"): "nginx.service - failed (Result: exit-code) since 04:31 UTC; nginx: [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)",
    # --- hard tier ---
    ("catalog-db-01", "disk_usage"): "Filesystem      Size  Used  Avail  Use%  Mounted on\n/dev/nvme0n1p2  500G  500G  0     100%  /var/lib/postgresql\nNote: pg_wal directory consuming 412G (WAL archiving stalled)",
    ("crm-node-01", "disk_usage"): "Filesystem  Size  Used  Avail  Use%  Mounted on\n/dev/sda1   200G  199G  1G     100%  /var\nLargest file: /var/log/crm_debug_trace.log (152G — debug logging left enabled)",
}

_TICKETS = {
    "INC-1234": {
        "status": "RESOLVED",
        "summary": "BigQuery nightly loads failing with quotaExceeded on reservation etl-slots",
        "resolution": "Raised slot reservation 'etl-slots' from 500 to 800 slots; added alerting on slot utilization > 85%",
    },
    "INC-2001": {"status": "OPEN", "summary": "Intermittent 502s on payments-api behind LB", "resolution": None},
    # --- hard tier (scenarios authored externally; canned data added to match) ---
    "TKT-1122": {
        "status": "OPEN",
        "summary": "store.corp/catalog rendering blank page",
        "attached_log_trace": "catalog-api returning 500s. PostgreSQL FATAL: could not write to WAL: No space left on device (host: catalog-db-01). Recommend checking disk usage on catalog-db-01.",
        "resolution": None,
    },
    "TKT-4450": {
        "status": "OPEN",
        "summary": "Nightly batch failure — scheduler reports BigQuery job 'sync_crm_to_bq_88' errored. Investigate root cause.",
        "resolution": None,
    },
    "TKT-9912": {
        "status": "OPEN",
        "summary": "CEO (ceo@enterprise.com) gets blank 'Access Denied' screen on the executive dashboard from her new personal mobile device. Works fine from her office laptop and for all other users. Suspect account/device policy — check her IdP account.",
        "resolution": None,
    },
}

# ---------------------------------------------------------------------------


def execute_tool(name: str, args: dict) -> str:
    """Execute a mock tool call; always returns a JSON string (errors included)."""
    try:
        result = _dispatch(name, {k: (v.strip() if isinstance(v, str) else v) for k, v in args.items()})
    except Exception as e:  # malformed args from the model are data, not crashes
        result = {"error": f"tool execution failed: {e}"}
    return json.dumps(result)


def _dispatch(name: str, args: dict):
    if name == "bq_job_status":
        job = _BQ_JOBS.get(args.get("job_id", ""))
        return {"job_id": args.get("job_id"), **job} if job else {"error": "job not found"}
    if name == "cloud_observability":
        m = _METRICS.get((args.get("service", ""), args.get("metric", "")))
        return {"service": args.get("service"), "metric": args.get("metric"), "window": args.get("window", "1h"), **m} if m else {"error": "no data for that service/metric"}
    if name == "k8s_pod_status":
        p = _PODS.get((args.get("namespace", ""), args.get("deployment", "")))
        return p or {"error": "deployment not found"}
    if name == "browser_console_logs":
        logs = _CONSOLE_LOGS.get(args.get("page_url", ""))
        return {"page_url": args.get("page_url"), "entries": logs} if logs else {"page_url": args.get("page_url"), "entries": []}
    if name == "lighthouse_audit":
        a = _LIGHTHOUSE.get((args.get("page_url", ""), args.get("category", "")))
        return a or {"error": "no audit data for that page/category"}
    if name == "api_probe":
        r = _API.get((args.get("endpoint", ""), args.get("method", "GET")))
        return r or {"error": "endpoint not reachable in test harness"}
    if name == "idp_user_lookup":
        u = _IDP_USERS.get(args.get("email", "").lower())
        return {"email": args.get("email"), **u} if u else {"error": "user not found"}
    if name == "host_command":
        if args.get("command") == "service_status":
            out = _HOSTS.get(((args.get("host", ""), "service_status"), args.get("service", "")))
        else:
            out = _HOSTS.get((args.get("host", ""), args.get("command", "")))
        return {"host": args.get("host"), "command": args.get("command"), "output": out} if out else {"error": "host or command not available"}
    if name == "ticket_lookup":
        t = _TICKETS.get(args.get("ticket_id", "").upper())
        return {"ticket_id": args.get("ticket_id"), **t} if t else {"error": "ticket not found"}
    return {"error": f"unknown tool: {name}"}
