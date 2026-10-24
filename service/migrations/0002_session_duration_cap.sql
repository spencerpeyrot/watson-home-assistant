ALTER TABLE sessions
    ADD COLUMN max_session_seconds INTEGER
    CHECK (max_session_seconds IS NULL OR max_session_seconds > 0);

INSERT INTO schema_migrations (version) VALUES ('0002_session_duration_cap')
    ON CONFLICT (version) DO NOTHING;
