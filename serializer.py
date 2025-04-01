import numpy as np
import os

def save_board(board_state, filename="board_save.npy"):
    """Speichert den Zustand des Spielbretts in einer Datei."""
    try:
        np.save(filename, board_state)
        print(f"Spielbrett erfolgreich in '{filename}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Spielbretts: {e}")

def load_board(filename="board_save.npy"):
    """Lädt den Zustand des Spielbretts aus einer Datei."""
    if os.path.exists(filename):
        try:
            board_state = np.load(filename)
            print(f"Spielbrett erfolgreich aus '{filename}' geladen.")
            # Optional: Überprüfe Dimensionen oder Typ, falls nötig
            if board_state.shape == (8, 8):
                return board_state
            else:
                print(f"Fehler: Geladene Matrix hat falsche Dimensionen {board_state.shape}. Erstelle Standardbrett.")
                return np.zeros((8, 8), dtype=int)
        except Exception as e:
            print(f"Fehler beim Laden des Spielbretts aus '{filename}': {e}. Erstelle Standardbrett.")
            return np.zeros((8, 8), dtype=int)
    else:
        print(f"Keine Speicherdatei '{filename}' gefunden. Erstelle Standardbrett.")
        return np.zeros((8, 8), dtype=int) 