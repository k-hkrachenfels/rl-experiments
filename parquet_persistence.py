import pyarrow as pa
import pyarrow.parquet as pq



# --- Ebene 4: Ein einzelner Schritt (Step) ---
step_struct = pa.struct([
    ('pos_x', pa.int16()),
    ('pos_y', pa.int16()),
    # Dictionary encoding für Actions ist effizient
    ('action', pa.dictionary(pa.int8(), pa.string())), 
    ('reward_to_go', pa.float32())
])

# --- Ebene 3: Die Welt (World) und die Episode ---

# Das Grid: Eine Liste von Listen von Strings (2D Array)
# Besser für ML wäre oft int8 (0=Wall, 1=Free...), aber hier als String wie gewünscht:
grid_type = pa.list_(pa.list_(pa.dictionary(pa.int8(), pa.string())))

world_struct = pa.struct([
    ('size_x', pa.int32()),
    ('size_y', pa.int32()),
    ('grid', grid_type)
])

episode_struct = pa.struct([
    ('nr', pa.int32()),
    # Eine Liste von Schritten
    ('steps', pa.list_(step_struct)) 
])

# --- Ebene 2: Das Experiment ---
experiment_struct = pa.struct([
    ('world', world_struct),
    ('episode', episode_struct)
])

# --- Ebene 1: Das Parquet Schema (Die Tabelle hat eine Hauptspalte 'experiment') ---
COMPLEX_SCHEMA = pa.schema([
    ('experiment', experiment_struct)
])

data = {
    'experiment': {
        'world': {
            'size_x': 3,
            'size_y': 3,
            # 3x3 Grid Beispiel
            'grid': [
                ['wall',  'wall',  'wall'],
                ['start', 'free',  'target'],
                ['wall',  'wall',  'wall']
            ]
        },
        'episode': {
            'nr': 101,
            'steps': [
                {'sequ_num': 0, 'pos_x': 0, 'pos_y': 1, 'action': 'RIGHT', 'reward_to_go': 5.0},
                {'sequ_num': 1, 'pos_x': 1, 'pos_y': 1, 'action': 'RIGHT', 'reward_to_go': 10.0},
                {'sequ_num': 2, 'pos_x': 2, 'pos_y': 1, 'action': 'STOP',  'reward_to_go': 0.0}
            ]
        }
    }
}

# Erstellen der Tabelle
# Wir müssen die Daten in eine Liste packen [], da Parquet spaltenbasiert ist (hier 1 Zeile)
table = pa.Table.from_pydict({
    'experiment': [data['experiment']]
}, schema=COMPLEX_SCHEMA)

# Schreiben
pq.write_table(table, "deep_nested_experiment.parquet")