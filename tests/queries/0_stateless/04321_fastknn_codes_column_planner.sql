-- fastknn codes-as-column: the query planner automatically rewrites ORDER BY distance LIMIT into a two-stage
-- shortlist (over the small quantized codes) + rescore (against the full-precision vector), reading the heavy vector
-- column lazily for the shortlisted rows only. Triggered purely by the presence of a self-describing MATERIALIZED
-- `fastknnEncode(...)` column - no index involved.

DROP TABLE IF EXISTS fastknn_auto;
CREATE TABLE fastknn_auto
(
    id UInt32,
    vec Array(Float32),
    code MATERIALIZED fastknnEncode(vec, 'rabitq', 64, 0)
)
ENGINE = MergeTree ORDER BY id;

INSERT INTO fastknn_auto (id, vec)
SELECT number, arrayMap(j -> toFloat32(sipHash64(number, j) % 2000 / 1000.0 - 1.0), range(64))
FROM numbers(4000);

-- The plan contains the inner fastknn shortlist and a lazy read of the vector column.
SELECT
    countIf(explain ILIKE '%fastknn shortlist%') > 0 AS has_shortlist,
    countIf(explain ILIKE '%LazilyReadFromMergeTree%') > 0 AS has_lazy_read
FROM
(
    EXPLAIN PLAN
    SELECT id FROM fastknn_auto
    ORDER BY L2Distance(vec, (SELECT vec FROM fastknn_auto WHERE id = 123)) ASC
    LIMIT 5 SETTINGS vector_search_index_fetch_multiplier = 50
);

-- With a shortlist that covers all rows, the rescore is exact: the codes path reproduces the exact brute-force top-k.
WITH (SELECT vec FROM fastknn_auto WHERE id = 123) AS ref
SELECT 'unfiltered top-10 exact==codes',
    (SELECT groupArray(id) FROM (SELECT id, L2Distance(vec, ref) AS d FROM fastknn_auto ORDER BY d, id LIMIT 10))
    = (SELECT groupArray(id) FROM (SELECT id FROM fastknn_auto ORDER BY L2Distance(vec, ref) ASC LIMIT 10 SETTINGS vector_search_index_fetch_multiplier = 4000));

-- Same with a post-filter (the original motivation): the WHERE is prefiltered before the shortlist.
WITH (SELECT vec FROM fastknn_auto WHERE id = 123) AS ref
SELECT 'filtered top-8 exact==codes',
    (SELECT groupArray(id) FROM (SELECT id, L2Distance(vec, ref) AS d FROM fastknn_auto WHERE id % 7 = 0 ORDER BY d, id LIMIT 8))
    = (SELECT groupArray(id) FROM (SELECT id FROM fastknn_auto WHERE id % 7 = 0 ORDER BY L2Distance(vec, ref) ASC LIMIT 8 SETTINGS vector_search_index_fetch_multiplier = 4000));

-- A WHERE that is NOT moved to PREWHERE stays a FilterStep; the shortlist is spliced ABOVE it so it still
-- prefilters. The rewrite fires (shortlist present) and the result is exact with a full shortlist.
SELECT
    countIf(explain ILIKE '%fastknn shortlist%') > 0 AS has_shortlist,
    countIf(explain ILIKE '%Filter%') > 0 AS has_filter_step
FROM
(
    EXPLAIN PLAN
    SELECT id FROM fastknn_auto WHERE id % 7 = 0
    ORDER BY L2Distance(vec, (SELECT vec FROM fastknn_auto WHERE id = 123)) ASC
    LIMIT 5 SETTINGS vector_search_index_fetch_multiplier = 50, optimize_move_to_prewhere = 0
);

WITH (SELECT vec FROM fastknn_auto WHERE id = 123) AS ref
SELECT 'filtered (FilterStep) top-8 exact==codes',
    (SELECT groupArray(id) FROM (SELECT id, L2Distance(vec, ref) AS d FROM fastknn_auto WHERE id % 7 = 0 ORDER BY d, id LIMIT 8))
    = (SELECT groupArray(id) FROM (SELECT id FROM fastknn_auto WHERE id % 7 = 0 ORDER BY L2Distance(vec, ref) ASC LIMIT 8 SETTINGS vector_search_index_fetch_multiplier = 4000, optimize_move_to_prewhere = 0));

-- The exact-match query vector is always returned first (its quantized code is the closest, rescore distance 0).
WITH (SELECT vec FROM fastknn_auto WHERE id = 123) AS ref
SELECT 'nearest is self', (SELECT id FROM fastknn_auto ORDER BY L2Distance(vec, ref) ASC LIMIT 1 SETTINGS vector_search_index_fetch_multiplier = 100) = 123;

DROP TABLE fastknn_auto;
