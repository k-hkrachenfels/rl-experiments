
SELECT 
    experiment.episode.nr,
    step.num,
    step.pos_x,
    step.pos_y,
    step.action,
    step.reward_to_go
FROM 'experiments/experiment_000100.parquet',
UNNEST (experiment.episode.steps) AS t(step);


SELECT 
    row_number() OVER () - 1 as row_idx,
    q_row
FROM 'experiments/experiment_000100.parquet',
UNNEST(experiment.q_table) AS t(q_row);
