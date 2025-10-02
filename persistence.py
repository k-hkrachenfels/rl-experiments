import numpy as np
import os
import glob
from config import Config

def save_board(config: Config, board_state):
    """
    Save the state of the game board to a file.
    """
    try:
        filepath = os.path.join(config.files.experiment_dir, config.files.board)
        np.save(filepath, board_state)
        print(f"Spielbrett erfolgreich in '{filepath}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Spielbretts: {e}")

def load_board(config: Config):
    """
    Load the state of the game board from a file.
    """
    filepath = os.path.join(config.files.experiment_dir, config.files.board)
    if os.path.exists(filepath):
        try:
            board_state = np.load(filepath)
            print(f"Spielbrett erfolgreich aus '{filepath}' geladen.")
            if board_state.shape == (config.world.grid_rows, config.world.grid_cols):
                return board_state
            else:
                print(f"Fehler: Geladene Matrix hat falsche Dimensionen {board_state.shape}. Erstelle Standardbrett.")
                return np.zeros((config.world.grid_rows, config.world.grid_cols), dtype=int)
        except Exception as e:
            print(f"Fehler beim Laden des Spielbretts aus '{filepath}': {e}. Erstelle Standardbrett.")
            return np.zeros((config.world.grid_rows, config.world.grid_cols), dtype=int)
    else:
        print(f"Keine Speicherdatei '{filepath}' gefunden. Erstelle Standardbrett.")
        return np.zeros((config.world.grid_rows, config.world.grid_cols), dtype=int)

def load_board_and_find_start_goal(config: Config):
    """Lädt das Board und findet Start/Ziel."""
    board = load_board(config)
    green_cells = np.argwhere(board == 3) # 3 is GREEN
    if len(green_cells) != 1:
        raise ValueError("Es muss genau ein grünes Start-Feld auf dem Board gefunden werden.")
    
    red_cells = np.argwhere(board == 4) # 4 is RED
    if len(red_cells) != 1:
        raise ValueError("Es muss genau ein rotes Ziel-Feld auf dem Board gefunden werden.")

    start = tuple(green_cells[0])
    goal = tuple(red_cells[0])
    return board, start, goal

def save_results(config: Config, board, agent, visualization_path, start_pos, goal_pos, episode_number):
    # Erstelle und speichere das Board mit dem Pfad
    board_with_path = board.copy()
    PATH_MARKER = 6  # Neue Konstante für den Pfad
    # Verwende visualization_path für die Markierung auf dem Board
    for r, c in visualization_path:
        # Markiere Pfad nur auf erlaubten Feldern (Weiß, Grün), nicht Start/Ziel selbst
        if (r, c) != start_pos and (r, c) != goal_pos and board_with_path[r, c] in [0, 3]: # 0 is WHITE, 3 is GREEN
            board_with_path[r, c] = PATH_MARKER


    board_with_path_file = f'{config.files.output_prefix}_{episode_number+1:06d}.npy'
    filepath = os.path.join(config.files.experiment_dir, board_with_path_file)
    np.save(filepath, board_with_path)

    # Speichere die Q-Tabelle
    try:
        q_table_to_save = agent.q_table

        # Überprüfe die Dimensionen (sollte 16, 16, 4 sein)
        if q_table_to_save.shape != (config.world.grid_rows, config.world.grid_cols, 4):
            print(f"Warnung: Unerwartete Q-Tabellen-Dimension beim Speichern: {q_table_to_save.shape}")

        q_table_file = f'{config.files.q_table_prefix}_{episode_number+1:06d}.npy'
        filepath = os.path.join(config.files.experiment_dir, q_table_file)
        np.save(filepath, q_table_to_save)
    except Exception as e:
        print(f"Fehler beim Speichern der Q-Tabelle: {e}")

# --- Q-Tabelle Laden/Speichern ---

def save_q_table(config: Config, q_table, episode_number):
    """
    Save the Q-table to a file.
    """
    try:
        q_table_file = f'{config.files.q_table_prefix}_{episode_number+1:06d}.npy'
        filepath = os.path.join(config.files.experiment_dir, q_table_file)
        np.save(filepath, q_table)
        print(f"Q-Tabelle erfolgreich in '{filepath}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Q-Tabelle: {e}")

def load_q_table(config: Config, episode_number):
    """
    Load the Q-table from a file.
    """
    q_table_file = f'{config.files.q_table_prefix}_{episode_number+1:06d}.npy'
    filepath = os.path.join(config.files.experiment_dir, q_table_file)
    if os.path.exists(filepath):
        try:
            q_table = np.load(filepath)
            print(f"Q-Tabelle erfolgreich aus '{filepath}' geladen.")
            return q_table
        except Exception as e:
            print(f"Fehler beim Laden der Q-Tabelle aus '{filepath}': {e}.")
            return None
    else:
        print(f"Keine Speicherdatei '{filepath}' gefunden.")
        return None

# --- Pfad Laden/Speichern ---

def save_path(config: Config, board_with_path, episode_number):
    """
    Save the board with the path to a file.
    """
    try:
        board_with_path_file = f'{config.files.output_prefix}_{episode_number+1:06d}.npy'
        filepath = os.path.join(config.files.experiment_dir, board_with_path_file)
        np.save(filepath, board_with_path)
        print(f"Board mit Pfad erfolgreich in '{filepath}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Boards mit Pfad: {e}")


def load_all_path_files(config: Config):
    """Lädt alle Board-Dateien mit Pfad und sortiert sie nach Episodennummer."""
    path_pattern = os.path.join(config.files.experiment_dir, f'{config.files.output_prefix}_*.npy')
    q_pattern = os.path.join(config.files.experiment_dir, f'{config.files.q_table_prefix}_*.npy')
    path_files = glob.glob(path_pattern)
    q_files = glob.glob(q_pattern)
    
    episodes = []
    for file in path_files:
        try:
            episode_num = int(file.split('_')[-1].split('.')[0])
            episodes.append((episode_num, file))
        except (ValueError, IndexError):
            print(f"Ungültiger Dateiname: {file}")
            continue
    
    episodes.sort()
    
    episode_files = {}
    for episode_num, board_file in episodes:
        q_file = board_file.replace(config.files.output_prefix, config.files.q_table_prefix)
        if q_file in q_files:
            episode_files[episode_num] = (board_file, q_file)
        else:
            print(f"Warnung: Keine passende Q-Tabelle gefunden für Episode {episode_num}")
    
    return episode_files

def load_editor_viewer_data(config: Config):
    """
    Lädt Board und Q-Tabelle für Editor/Viewer (16x16).
    Prioritisiert Pfad-Datei für initialen Board-View.
    """
    board_state = None
    path_loaded = False
    q_table = None
    path_files = load_all_path_files(config)

    if board_state is None:
        board_state = load_board(config)
        
    if path_files:
        print(f"Gefundene Pfad-Dateien: {len(path_files)}")
        for episode_num, file in path_files.items():
            print(f"  Episode {episode_num}: {file}")
        
        first_episode = min(path_files.keys())
        first_path_file, first_q_file = path_files[first_episode]
        print(f"\nLade erste Pfad-Datei: {first_path_file}")
        try:
            loaded_board = np.load(first_path_file)
            if loaded_board.shape == (config.world.grid_rows, config.world.grid_cols):
                print("Pfad-Board geladen.")
                board_state = loaded_board
                path_loaded = True
                try:
                    q_table = np.load(first_q_file)
                    print(f"Q-Tabelle für Episode {first_episode} geladen")
                except Exception as e:
                    print(f"Fehler beim Laden der Q-Tabelle: {e}")
            else:
                print(f"Warnung: Pfad-Board hat falsche Dimensionen ({loaded_board.shape}). Ignoriere.")
        except Exception as e:
            print(f"Fehler beim Laden von {first_path_file}: {e}. Ignoriere.")



    return board_state, path_loaded, q_table, path_files

def load_path_file(config: Config, file_path):
    """Lädt eine Pfad-Datei und aktualisiert das Board."""
    try:
        loaded_board = np.load(file_path)
        if loaded_board.shape == (config.world.grid_rows, config.world.grid_cols):
            # Erstelle eine Kopie des Original-Boards
            original_board = load_board(config)
            # Setze nur die Pfadmarker (6) auf dem Original-Board
            board_state = np.where(loaded_board == 6, 6, original_board)
            return board_state
        else:
            print(f"Warnung: Pfad-Board hat falsche Dimensionen ({loaded_board.shape}).")
            return None
    except Exception as e:
        print(f"Fehler beim Laden von {file_path}: {e}")
        return None

def delete_old_files(config: Config):
    """Löscht alle alten Board-Dateien und q-value Dateien mit Pfad aus vorherigen Episoden."""
    path_pattern = os.path.join(config.files.experiment_dir, f'{config.files.output_prefix}_*.npy')
    q_pattern = os.path.join(config.files.experiment_dir, f'{config.files.q_table_prefix}_*.npy')
    patterns=[path_pattern, q_pattern]
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Fehler beim Löschen von {file}: {e}")


