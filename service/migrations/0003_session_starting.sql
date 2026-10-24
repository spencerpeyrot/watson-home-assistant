ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_state_check;

ALTER TABLE sessions
    ADD CONSTRAINT sessions_state_check
    CHECK (state IN ('connecting', 'starting', 'ready', 'closing', 'closed'));

INSERT INTO schema_migrations (version) VALUES ('0003_session_starting')
    ON CONFLICT (version) DO NOTHING;
