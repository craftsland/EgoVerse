#!/usr/bin/env python3
"""Scan a processed_v3/<folder> tree on S3 and stage every zarr's metadata into a
SQL staging table (default ``app.staging_<folder>``) for review before merging
into ``app.episodes``.

For each ``*.zarr`` episode found (recursively — some folders nest by category,
e.g. mecka/flagship/), it reads the zarr group metadata (``zarr.json`` /
``.zattrs``) and stages one row with the columns that map to ``app.episodes``
(``episode_hash``, ``embodiment``, ``task``, ``task_description``,
``num_frames``, ``zarr_processed_path``) plus staging-only fields (``fps``,
``has_annotations``, ``source_folder``, ``created_at`` parsed from the hash when
it is a recording timestamp, ``raw_attrs`` = the full attributes as JSONB, and
``scanned_at``). ``episode_hash`` = the ``.zarr`` directory name; the table is a
PRIMARY KEY on it, so duplicate hashes within the scan are rejected.

The staging table is rebuilt from scratch on each run (DROP + CREATE). It does
NOT touch app.episodes — that is the separate merge step.

Usage:
    python -m egomimic.scripts.backfill_scripts.stage_processed_folder \
        --folder microagi [--staging-table app.staging_microagi] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import text

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client
from egomimic.utils.aws.aws_sql import create_default_engine

BUCKET = "rldb"
_TS_HASH = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}$")


def find_zarr_prefixes(s3, base):
    """Recursively yield every ``<...>.zarr/`` prefix under ``base``."""
    stack = [base]
    while stack:
        prefix = stack.pop()
        token = None
        while True:
            kw = dict(Bucket=BUCKET, Prefix=prefix, Delimiter="/", MaxKeys=1000)
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for cp in resp.get("CommonPrefixes", []):
                sub = cp["Prefix"]
                if sub.rstrip("/").endswith(".zarr"):
                    yield sub
                else:
                    stack.append(sub)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")


def read_attrs(s3, zarr_prefix):
    """Return the zarr group attributes dict, or {} if unreadable."""
    for meta_name in ("zarr.json", ".zattrs"):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=zarr_prefix + meta_name)["Body"].read()
        except Exception:
            continue
        meta = json.loads(body)
        return meta.get("attributes", meta) or {}
    return {}


def created_at_from_hash(episode_hash):
    if not _TS_HASH.match(episode_hash):
        return None
    try:
        return datetime.strptime(episode_hash, "%Y-%m-%d-%H-%M-%S-%f").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True,
                    help="Subfolder under processed_v3/ to scan (e.g. microagi).")
    ap.add_argument("--staging-table", default=None,
                    help="Fully-qualified staging table (default app.staging_<folder>).")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N zarrs (testing).")
    ap.add_argument("--workers", type=int, default=48,
                    help="Parallel threads for reading zarr metadata (default 48).")
    args = ap.parse_args()

    folder = args.folder.strip("/")
    table = args.staging_table or (
        "app.staging_" + re.sub(r"[^a-z0-9_]", "_", folder.lower())
    )
    base = f"processed_v3/{folder}/"

    s3 = get_boto3_s3_client()
    engine = create_default_engine()

    # 1) list all episode prefixes (paginated listing is cheap: 1000/call).
    prefixes = list(find_zarr_prefixes(s3, base))
    if args.limit is not None:
        prefixes = prefixes[: args.limit]
    print(f"Found {len(prefixes)} .zarr prefixes; reading metadata "
          f"({args.workers} threads)...", flush=True)

    # 2) read each zarr's metadata in parallel — the per-episode get_object is
    #    network-bound, so threading it is the difference between minutes and
    #    seconds. boto3 clients are thread-safe.
    def build_row(zp):
        episode_hash = zp.rstrip("/").split("/")[-1][: -len(".zarr")]
        a = read_attrs(s3, zp)
        feats = a.get("features")
        return {
            "episode_hash": episode_hash,
            "embodiment": a.get("embodiment"),
            "task": a.get("task_name"),
            "task_description": a.get("task_description"),
            "num_frames": a.get("total_frames"),
            "zarr_processed_path": f"s3://{BUCKET}/{zp.rstrip('/')}",
            "fps": a.get("fps"),
            "has_annotations": isinstance(feats, dict) and "annotations" in feats,
            "source_folder": folder,
            "created_at": created_at_from_hash(episode_hash),
            "raw_attrs": json.dumps(a),
        }

    rows, seen, dupes, done = [], set(), [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(build_row, prefixes):
            done += 1
            if row["episode_hash"] in seen:
                dupes.append(row["episode_hash"])
                continue
            seen.add(row["episode_hash"])
            rows.append(row)
            if done % 500 == 0:
                print(f"  read {done}/{len(prefixes)}...", flush=True)

    print(f"Scanned {len(rows)} unique zarr episodes under s3://{BUCKET}/{base}")
    if dupes:
        print(f"  WARNING: {len(dupes)} duplicate episode_hash(es) skipped: {dupes[:5]}")

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text(f"""
            CREATE TABLE {table} (
                episode_hash        TEXT PRIMARY KEY,
                embodiment          TEXT,
                task                TEXT,
                task_description    TEXT,
                num_frames          INTEGER,
                zarr_processed_path TEXT,
                fps                 DOUBLE PRECISION,
                has_annotations     BOOLEAN,
                source_folder       TEXT,
                created_at          TIMESTAMPTZ,
                raw_attrs           JSONB,
                scanned_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        if rows:
            conn.execute(text(f"""
                INSERT INTO {table}
                    (episode_hash, embodiment, task, task_description, num_frames,
                     zarr_processed_path, fps, has_annotations, source_folder,
                     created_at, raw_attrs)
                VALUES
                    (:episode_hash, :embodiment, :task, :task_description, :num_frames,
                     :zarr_processed_path, :fps, :has_annotations, :source_folder,
                     :created_at, CAST(:raw_attrs AS JSONB))
            """), rows)

    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        fill = conn.execute(text(f"""
            SELECT
              COUNT(*) FILTER (WHERE embodiment IS NOT NULL)          AS embodiment,
              COUNT(*) FILTER (WHERE task IS NOT NULL)                AS task,
              COUNT(*) FILTER (WHERE task_description IS NOT NULL)    AS task_description,
              COUNT(*) FILTER (WHERE num_frames IS NOT NULL)         AS num_frames,
              COUNT(*) FILTER (WHERE zarr_processed_path IS NOT NULL) AS zarr_path,
              COUNT(*) FILTER (WHERE created_at IS NOT NULL)          AS created_at,
              COUNT(*) FILTER (WHERE has_annotations)                AS has_annotations
            FROM {table}
        """)).mappings().first()
        sample = conn.execute(text(f"""
            SELECT episode_hash, embodiment, task, num_frames, created_at, has_annotations
            FROM {table} ORDER BY episode_hash LIMIT 5
        """)).all()

    print(f"\nStaged {n} rows into {table}")
    print("  column fill counts (of {} rows):".format(n))
    for k, v in fill.items():
        print(f"    {k:20} {v}")
    print("  sample rows:")
    for s in sample:
        print(f"    {s.episode_hash}  emb={s.embodiment} task={s.task!r} "
              f"nf={s.num_frames} created_at={s.created_at} ann={s.has_annotations}")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
