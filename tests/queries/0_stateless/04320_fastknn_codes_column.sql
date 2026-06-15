-- fastknnEncode / fastknnDistance: codes-as-column foundation.

SELECT 'byte lengths per method';
SELECT
    length(fastknnEncode(materialize(arrayMap(x -> toFloat32(x), range(64))), 'b1', 64, 0)) AS b1,
    length(fastknnEncode(materialize(arrayMap(x -> toFloat32(x), range(64))), 'b1_projected', 64, 0)) AS b1p,
    length(fastknnEncode(materialize(arrayMap(x -> toFloat32(x), range(64))), 'rabitq', 64, 0)) AS rabitq,
    length(fastknnEncode(materialize(arrayMap(x -> toFloat32(x), range(64))), 'turboquant', 64, 0)) AS turboquant,
    length(fastknnEncode(materialize(arrayMap(x -> toFloat32(x), range(64))), 'e8', 64, 8)) AS e8;

-- A MATERIALIZED code column is populated on INSERT.
DROP TABLE IF EXISTS fastknn_codes;
CREATE TABLE fastknn_codes
(
    id UInt32,
    vec Array(Float32),
    code FixedString(8) MATERIALIZED fastknnEncode(vec, 'b1', 64, 0)
)
ENGINE = MergeTree ORDER BY id;

-- 8 well-separated signed vectors: vector i is +1 in its own 8-dim block and -1 elsewhere.
INSERT INTO fastknn_codes (id, vec)
SELECT
    number AS id,
    arrayMap(x -> toFloat32(if(intDiv(x, 8) = number % 8, 1, -1)), range(64)) AS vec
FROM numbers(8);

SELECT 'materialized code is non-empty and fixed size';
SELECT countDistinct(code) >= 1, any(length(code)) FROM fastknn_codes;

-- The approximate nearest neighbour of axis-3 query should be id=3 (the matching signed vector).
SELECT 'nearest by approximate distance (b1)';
SELECT id
FROM fastknn_codes
ORDER BY fastknnDistance(code, arrayMap(x -> toFloat32(if(intDiv(x, 8) = 3, 1, -1)), range(64)), 'b1', 64, 0, 1) ASC, id ASC
LIMIT 1;

-- Prefilter + rank + rescore pattern: filter, rank by approximate distance, then rescore against full precision.
SELECT 'prefilter then rescore (top-2 reranked by exact L2)';
SELECT id
FROM
(
    SELECT id, vec
    FROM fastknn_codes
    WHERE id != 0                                   -- post-filter predicate
    ORDER BY fastknnDistance(code, arrayMap(x -> toFloat32(if(intDiv(x, 8) = 5, 1, -1)), range(64)), 'b1', 64, 0, 1) ASC, id ASC
    LIMIT 4                                          -- shortlist by approximate distance
)
ORDER BY L2Distance(vec, arrayMap(x -> toFloat32(if(intDiv(x, 8) = 5, 1, -1)), range(64))) ASC, id ASC  -- rescore
LIMIT 2;

DROP TABLE fastknn_codes;

-- Error handling.
SELECT 'errors';
SELECT fastknnEncode([1., 2., 3.], 'nope', 3, 0); -- { serverError BAD_ARGUMENTS }
SELECT fastknnEncode(materialize([1., 2.]), 'b1', 8, 0); -- { serverError SIZES_OF_ARRAYS_DONT_MATCH }
