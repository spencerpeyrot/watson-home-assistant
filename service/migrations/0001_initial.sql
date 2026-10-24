-- Cube Assistant application data. LangGraph owns its checkpoint tables;
-- this migration owns only application records and their access boundaries.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE conversations (
    conversation_id UUID NOT NULL,
    owner_id UUID NOT NULL,
    device_id TEXT NOT NULL CHECK (length(device_id) > 0),
    locale TEXT NOT NULL CHECK (length(locale) >= 2),
    timezone TEXT NOT NULL CHECK (length(timezone) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, owner_id)
);

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('connecting', 'ready', 'closing', 'closed')),
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    idle_close_seconds INTEGER NOT NULL CHECK (idle_close_seconds >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, owner_id),
    FOREIGN KEY (conversation_id, owner_id)
        REFERENCES conversations (conversation_id, owner_id),
    CHECK (closed_at IS NULL OR closed_at >= opened_at)
);

CREATE INDEX sessions_owner_opened_idx ON sessions (owner_id, opened_at DESC);

CREATE TABLE jobs (
    job_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    device_id TEXT NOT NULL CHECK (length(device_id) > 0),
    conversation_id UUID NOT NULL,
    session_id UUID NOT NULL,
    turn_id UUID NOT NULL,
    delegation_id TEXT NOT NULL CHECK (length(delegation_id) > 0),
    trace_id UUID NOT NULL,
    parent_job_id UUID,
    context_revision INTEGER NOT NULL DEFAULT 0 CHECK (context_revision >= 0),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'waiting_for_approval', 'succeeded',
                   'failed', 'cancelled', 'needs_reconciliation')
    ),
    request_text TEXT NOT NULL CHECK (length(request_text) > 0),
    request JSONB NOT NULL CHECK (jsonb_typeof(request) = 'object'),
    request_digest CHAR(64) NOT NULL CHECK (request_digest ~ '^[a-f0-9]{64}$'),
    idempotency_key TEXT NOT NULL CHECK (length(idempotency_key) BETWEEN 1 AND 200),
    summary TEXT NOT NULL CHECK (length(summary) > 0),
    last_sequence BIGINT NOT NULL DEFAULT 0 CHECK (last_sequence >= 0),
    snapshot JSONB NOT NULL CHECK (jsonb_typeof(snapshot) = 'object'),
    claimed_by UUID,
    lease_expires_at TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, owner_id),
    UNIQUE (owner_id, idempotency_key),
    UNIQUE (owner_id, session_id, delegation_id),
    FOREIGN KEY (conversation_id, owner_id)
        REFERENCES conversations (conversation_id, owner_id),
    FOREIGN KEY (session_id, owner_id)
        REFERENCES sessions (session_id, owner_id),
    FOREIGN KEY (parent_job_id, owner_id)
        REFERENCES jobs (job_id, owner_id),
    CHECK ((claimed_by IS NULL) = (lease_expires_at IS NULL)),
    CHECK (updated_at >= created_at)
);

CREATE INDEX jobs_owner_created_idx ON jobs (owner_id, created_at DESC);
CREATE INDEX jobs_owner_status_updated_idx ON jobs (owner_id, status, updated_at DESC);
CREATE INDEX jobs_claimable_idx ON jobs (status, lease_expires_at, created_at)
    WHERE status IN ('queued', 'running');

CREATE TABLE job_events (
    event_id UUID PRIMARY KEY,
    job_id UUID NOT NULL,
    owner_id UUID NOT NULL,
    sequence BIGINT NOT NULL CHECK (sequence >= 1),
    kind TEXT NOT NULL CHECK (
        kind IN ('status', 'progress', 'correction_acknowledged', 'approval_requested',
                 'artifact_ready', 'result')
    ),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'waiting_for_approval', 'succeeded',
                   'failed', 'cancelled', 'needs_reconciliation')
    ),
    context_revision INTEGER NOT NULL CHECK (context_revision >= 0),
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (job_id, owner_id, sequence),
    FOREIGN KEY (job_id, owner_id) REFERENCES jobs (job_id, owner_id)
);

CREATE INDEX job_events_owner_job_sequence_idx
    ON job_events (owner_id, job_id, sequence);

CREATE TABLE action_ledger (
    action_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    job_id UUID,
    trace_id UUID NOT NULL,
    tool_name TEXT NOT NULL CHECK (length(tool_name) > 0),
    resource_key TEXT NOT NULL CHECK (length(resource_key) > 0),
    payload JSONB NOT NULL,
    payload_digest CHAR(64) NOT NULL CHECK (payload_digest ~ '^[a-f0-9]{64}$'),
    idempotency_key TEXT NOT NULL CHECK (length(idempotency_key) > 0),
    state TEXT NOT NULL CHECK (
        state IN ('prepared', 'executing', 'succeeded', 'failed', 'uncertain',
                  'needs_reconciliation', 'denied')
    ),
    external_request_id TEXT,
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (action_id, owner_id),
    UNIQUE (owner_id, idempotency_key),
    FOREIGN KEY (job_id, owner_id) REFERENCES jobs (job_id, owner_id),
    CHECK (updated_at >= created_at)
);

CREATE INDEX action_ledger_owner_state_updated_idx
    ON action_ledger (owner_id, state, updated_at DESC);

CREATE TABLE approvals (
    approval_id UUID PRIMARY KEY,
    action_id UUID NOT NULL,
    owner_id UUID NOT NULL,
    payload_digest CHAR(64) NOT NULL CHECK (payload_digest ~ '^[a-f0-9]{64}$'),
    context_revision INTEGER NOT NULL CHECK (context_revision >= 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'denied', 'expired')),
    expires_at TIMESTAMPTZ NOT NULL,
    decided_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (approval_id, owner_id),
    FOREIGN KEY (action_id, owner_id) REFERENCES action_ledger (action_id, owner_id),
    CHECK (decided_at IS NULL OR decided_at >= created_at)
);

CREATE INDEX approvals_owner_pending_expiry_idx
    ON approvals (owner_id, expires_at) WHERE status = 'pending';

CREATE TABLE memories (
    memory_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('preference', 'fact')),
    content TEXT NOT NULL CHECK (length(content) > 0),
    source_turn_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CHECK (updated_at >= created_at)
);

CREATE INDEX memories_owner_updated_idx ON memories (owner_id, updated_at DESC);

CREATE TABLE timers (
    timer_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    device_id TEXT NOT NULL CHECK (length(device_id) > 0),
    conversation_id UUID,
    due_at TIMESTAMPTZ NOT NULL,
    message TEXT NOT NULL CHECK (length(message) BETWEEN 1 AND 500),
    request_digest CHAR(64) NOT NULL CHECK (request_digest ~ '^[a-f0-9]{64}$'),
    status TEXT NOT NULL CHECK (
        status IN ('scheduled', 'firing', 'delivered', 'cancelled', 'failed')
    ),
    idempotency_key TEXT NOT NULL CHECK (length(idempotency_key) > 0),
    claimed_by UUID,
    lease_expires_at TIMESTAMPTZ,
    retry_after TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner_id, idempotency_key),
    FOREIGN KEY (conversation_id, owner_id)
        REFERENCES conversations (conversation_id, owner_id),
    CHECK ((claimed_by IS NULL) = (lease_expires_at IS NULL)),
    CHECK (delivered_at IS NULL OR delivered_at >= created_at)
);

CREATE INDEX timers_due_idx ON timers (COALESCE(retry_after, due_at), timer_id)
    WHERE status IN ('scheduled', 'firing');

CREATE TABLE artifacts (
    artifact_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    job_id UUID NOT NULL,
    label TEXT NOT NULL CHECK (length(label) > 0),
    media_type TEXT NOT NULL CHECK (length(media_type) > 0),
    storage_key TEXT NOT NULL UNIQUE,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    UNIQUE (artifact_id, owner_id),
    FOREIGN KEY (job_id, owner_id) REFERENCES jobs (job_id, owner_id),
    CHECK (deleted_at IS NULL OR deleted_at >= created_at)
);

CREATE INDEX artifacts_owner_created_idx ON artifacts (owner_id, created_at DESC)
    WHERE deleted_at IS NULL;

CREATE TABLE usage_measurements (
    usage_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    session_id UUID,
    job_id UUID,
    provider TEXT NOT NULL CHECK (length(provider) > 0),
    input_tokens BIGINT CHECK (input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens >= 0),
    connected_seconds NUMERIC(12, 3) CHECK (connected_seconds >= 0),
    cost_usd NUMERIC(12, 6) CHECK (cost_usd >= 0),
    cost_kind TEXT CHECK (cost_kind IN ('measured', 'estimated')),
    observed_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (session_id, owner_id) REFERENCES sessions (session_id, owner_id),
    FOREIGN KEY (job_id, owner_id) REFERENCES jobs (job_id, owner_id),
    CHECK ((cost_usd IS NULL) = (cost_kind IS NULL))
);

CREATE INDEX usage_owner_observed_idx
    ON usage_measurements (owner_id, observed_at DESC);

CREATE TABLE idempotency_records (
    owner_id UUID NOT NULL,
    operation TEXT NOT NULL CHECK (length(operation) > 0),
    idempotency_key TEXT NOT NULL CHECK (length(idempotency_key) > 0),
    request_digest CHAR(64) NOT NULL CHECK (request_digest ~ '^[a-f0-9]{64}$'),
    response JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (owner_id, operation, idempotency_key),
    CHECK (expires_at > created_at)
);

INSERT INTO schema_migrations (version) VALUES ('0001_initial')
    ON CONFLICT (version) DO NOTHING;
