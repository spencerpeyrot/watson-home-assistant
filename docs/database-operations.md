# Database backup and restore runbook

## Current state

PostgreSQL 17 now runs locally on the Cube Pi. The `cube-assistant` OS account matches a peer-authenticated login role and owns the `cube_assistant` database. `DATABASE_URL=postgresql:///cube_assistant?user=cube-assistant` works for processes running as that OS account. Initial application migrations `0001_initial`, `0002_session_duration_cap`, and `0003_session_starting` are applied. A disposable local PostgreSQL/artifact restore rehearsal passed on 2026-10-07. Backup destination, schedule, retention period, RPO, RTO, and encryption provider remain open. Do not treat a database dump as a backup of generated artifacts.

Before pilot, choose an encrypted off-Pi backup destination, encryption/key owner, schedule/retention, RPO/RTO, alert destination, and an operator authorized to restore. Keep the local database credential in a root-owned mode-0600 service environment file. Never put a password in shell history, command arguments, logs, or this repository.

## Backup

Use a custom-format logical dump. It is portable across minor PostgreSQL upgrades and can be restored into a separate candidate database.

```sh
umask 077
stamp=$(date -u +%Y%m%dT%H%M%SZ)
pg_dump --format=custom --no-owner --no-acl \
  --file="$BACKUP_DIR/cube-$stamp.dump" \
  --dbname="service=cube-backup"
pg_restore --list "$BACKUP_DIR/cube-$stamp.dump" >/dev/null
```

Set `PGSERVICEFILE` and `PGPASSFILE` in the service environment; keep both outside the repo and readable only by the service/operator account. Configure the libpq service as TLS-verified for remote PostgreSQL. Store the dump in an encrypted, access-controlled destination and retain its timestamp, application revision, PostgreSQL version, and checksum alongside the object.

Artifact metadata lives in PostgreSQL; artifact bytes live in the configured `ArtifactStore`. Back up both from a coordinated point. For local storage, take a filesystem snapshot or pause artifact cleanup while making a consistent copy. Preserve owner-partitioned opaque paths and verify each copied object's size and SHA-256 against metadata. An isolated DB dump without matching artifact bytes can restore broken download links.

## Restore drill

Restore to a **new, empty candidate database**. Never use `--clean`, `--create`, or an existing production database as the destination.

```sh
createdb --maintenance-db="service=cube-backup-admin" "$RESTORE_DB"
pg_restore --no-owner --no-acl \
  --dbname="$RESTORE_DB" \
  "$BACKUP_FILE"
```

Use a candidate deployment matching the recorded application revision. Run the repository migration runner against the candidate only, then verify operator health, owner isolation, recent session/job/usage queries, memory/timer records, artifact metadata, and artifact checksums/downloads. Exercise a small read-only request before accepting the restore. Keep the prior database and original backup untouched until an operator approves recovery completion.

Record the backup timestamp, database/application versions, restore start/end, verification results, missing artifacts, and measured recovery time. Repeat the drill after major migration or storage changes and at the owner-selected interval.

## Open release inputs

- Backup destination, encryption key custody, and access policy.
- Schedule and retention; RPO, RTO, alert path, and named restore operator.
- Artifact-store snapshot/replication method and matching retention.
- A restore drill against the Pi-local database and off-Pi artifact backup before pilot; the disposable local drill passed.

No production backup or production restore has run from this project workspace.
