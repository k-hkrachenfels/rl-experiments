
SELECT
    -- Wir generieren eine ID für die ursprüngliche Zeile (falls mehrere Experimente im File sind)
    row_number() OVER () as exp_id,
    -- Erste Ebene entpacken (Das Grid in Zeilen zerlegen)
    generate_subscripts(experiment.q_table, 1) as row_idx,
    q_rows
    
FROM 'experiments/experiment_000100.parquet',
unnest(experiment.q_table) as _(q_rows)
