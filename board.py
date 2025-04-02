import numpy as np
import pygame
import sys
import os
import math
import glob
# serializer wird nicht mehr direkt hier verwendet, da wir das Board mit Pfad separat laden
# from serializer import save_board, load_board

# --- Konstanten (aus sarsa_agent.py kopiert) ---
WHITE = 0
BLACK = 1
BLUE = 2
GREEN = 3
RED = 4
PATH_MARKER = 6 # Für den Pfad
GRAY = (128, 128, 128) # Farbe für den Pfad
NUM_COLORS = 5 # Anzahl der Farben zum Durchschalten
ARROW_COLOR = (100, 0, 0) # Dunkelrot für Pfeile
ARROW_THICKNESS = 2

# --- Board Laden/Speichern (vereinfacht, da sarsa_agent die Pfad-Datei erstellt) ---
BOARD_FILE_ORIGINAL = 'board_save.npy' # Original vom Editor
BOARD_FILE_WITH_PATH = 'board_with_path.npy' # Vom Agenten generiert
BOARD_FILE = 'board_save.npy'
Q_TABLE_FILE = 'q_table_final.npy'

def load_all_path_files():
    """
    Lädt alle verfügbaren Pfad-Dateien und sortiert sie nach Episodennummer.
    Returns: Liste von (episode_number, file_path) Tupeln
    """
    # Finde alle Pfad-Dateien
    path_files = glob.glob('board_with_path_*.npy')
    
    # Extrahiere Episodennummern und erstelle Tupel
    path_data = []
    for file in path_files:
        try:
            # Extrahiere die Nummer aus dem Dateinamen (board_with_path_000100.npy -> 100)
            episode_num = int(file.split('_')[-1].split('.')[0])
            path_data.append((episode_num, file))
        except (ValueError, IndexError):
            # Ignoriere Dateien, die nicht dem erwarteten Format entsprechen
            continue
    
    # Sortiere nach Episodennummer
    path_data.sort(key=lambda x: x[0])
    
    # Füge die finale Pfad-Datei hinzu, falls vorhanden
    if os.path.exists(BOARD_FILE_WITH_PATH):
        path_data.append((float('inf'), BOARD_FILE_WITH_PATH))  # Unendlich für finale Version
    
    return path_data

def load_editor_viewer_data():
    """
    Lädt Board und Q-Tabelle für Editor/Viewer (16x16).
    Prioritisiert Pfad-Datei für initialen Board-View.
    Returns: tuple (board_state, path_was_loaded, q_table, path_files)
    """
    board_state = None
    path_loaded = False
    q_table = None
    path_files = load_all_path_files()

    # 1. Versuche Pfad-Board zu laden
    if path_files:
        print(f"Gefundene Pfad-Dateien: {len(path_files)}")
        for episode_num, file in path_files:
            print(f"  Episode {episode_num}: {file}")
        
        # Lade die erste Pfad-Datei
        _, first_path_file = path_files[0]
        print(f"\nLade erste Pfad-Datei: {first_path_file}")
        try:
            loaded_board = np.load(first_path_file)
            if loaded_board.shape == (16, 16):
                print("Pfad-Board geladen.")
                board_state = loaded_board
                path_loaded = True
            else:
                print(f"Warnung: Pfad-Board hat falsche Dimensionen ({loaded_board.shape}). Ignoriere.")
        except Exception as e:
            print(f"Fehler beim Laden von {first_path_file}: {e}. Ignoriere.")

    # 2. Wenn Pfad nicht geladen, versuche normales Editor-Board
    if board_state is None and os.path.exists(BOARD_FILE):
        print(f"Lade Editor-Board: {BOARD_FILE}")
        try:
            loaded_board = np.load(BOARD_FILE)
            if loaded_board.shape == (16, 16):
                print("Editor-Board geladen.")
                board_state = loaded_board
            else:
                print(f"Warnung: Editor-Board hat falsche Dimensionen ({loaded_board.shape}). Erstelle neues Board.")
        except Exception as e:
            print(f"Fehler beim Laden von {BOARD_FILE}: {e}. Erstelle neues Board.")

    # 3. Wenn immer noch kein Board, erstelle neues
    if board_state is None:
        print("Keine gültige Board-Datei gefunden. Erstelle neues leeres 16x16 Board.")
        board_state = np.zeros((16, 16), dtype=int)

    # 4. Versuche Q-Tabelle zu laden
    if os.path.exists(Q_TABLE_FILE):
        print(f"Versuche Q-Tabelle zu laden: {Q_TABLE_FILE}")
        try:
            loaded_q = np.load(Q_TABLE_FILE)
            # Erwarte Shape (16, 16, 4)
            if loaded_q.shape == (16, 16, 4):
                q_table = loaded_q
                print("Q-Tabelle erfolgreich geladen.")
            else:
                print(f"Warnung: Geladene Q-Tabelle hat falsche Dimensionen ({loaded_q.shape}). Erwartet (16, 16, 4).")
        except Exception as e:
            print(f"Fehler beim Laden der Q-Tabelle ({Q_TABLE_FILE}): {e}")

    return board_state, path_loaded, q_table, path_files

def load_board():
    """Lädt das Board aus BOARD_FILE oder erstellt ein neues."""
    if os.path.exists(BOARD_FILE):
        print(f"Lade Board aus: {BOARD_FILE}")
        try:
            board = np.load(BOARD_FILE)
            # Stelle sicher, dass es 16x16 ist, sonst erstelle neu
            if board.shape == (16, 16):
                return board
            else:
                print(f"Warnung: Geladenes Board hat falsche Dimensionen ({board.shape}). Erwarte (16, 16). Erstelle neues Board.")
        except Exception as e:
            print(f"Fehler beim Laden von {BOARD_FILE}: {e}. Erstelle neues Board.")

    print("Keine gültige Board-Datei gefunden. Erstelle neues leeres 16x16 Board.")
    return np.zeros((16, 16), dtype=int) # Standard 16x16, weiß

def save_board(board_state):
    """Speichert das Board IMMER in die Standard-Editor-Datei BOARD_FILE."""
    try:
        # Stelle sicher, dass keine Pfadmarker gespeichert werden (sollte durch Editieren nicht passieren)
        # Sicherungshalber: Ersetze Werte > 4 durch 0 (Weiß)
        board_to_save = np.where(board_state >= NUM_COLORS, WHITE, board_state)
        np.save(BOARD_FILE, board_to_save)
        print(f"Board gespeichert in: {BOARD_FILE}")
    except Exception as e:
        print(f"Fehler beim Speichern von {BOARD_FILE}: {e}")

class BoardEditorViewer:
    def __init__(self):
        # Konstanten für das Brett (16x16)
        self.FIELD_SIZE = 30
        self.BOARD_ROWS = 16
        self.BOARD_COLS = 16
        self.BOARD_WIDTH = self.FIELD_SIZE * self.BOARD_COLS
        self.BOARD_HEIGHT = self.FIELD_SIZE * self.BOARD_ROWS
        self.MARGIN = 20
        self.BUTTON_HEIGHT = 40
        self.WINDOW_WIDTH = self.BOARD_WIDTH + 2*self.MARGIN
        self.WINDOW_HEIGHT = self.BOARD_HEIGHT + 2*self.MARGIN + 2*self.BUTTON_HEIGHT + 10  # Extra Höhe für zweiten Button

        # Initialisiere Pygame
        pygame.init()

        # Erstelle das Fenster
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Board Editor/Viewer mit Q-Werten (16x16)")

        # Lade initialen Zustand (priorisiert Pfad)
        self.board_state, self.path_initially_loaded, self.q_table, self.path_files = load_editor_viewer_data()
        self.current_path_index = 0 if self.path_files else -1
        
        if self.q_table is not None:
            print("Anzeige: Q-Werte werden visualisiert.")
        else:
            print("Anzeige: Keine Q-Werte gefunden/geladen.")

        # Erstelle die Buttons
        self.save_button_rect = pygame.Rect(
            self.MARGIN,
            self.BOARD_HEIGHT + 2*self.MARGIN,
            self.BOARD_WIDTH,
            self.BUTTON_HEIGHT
        )
        
        self.next_path_button_rect = pygame.Rect(
            self.MARGIN,
            self.BOARD_HEIGHT + 3*self.MARGIN + self.BUTTON_HEIGHT,
            self.BOARD_WIDTH,
            self.BUTTON_HEIGHT
        )

        # Farben-Mapping für die Anzeige (inkl. Pfad)
        self.colors = {
            WHITE: (255, 255, 255),
            BLACK: (0, 0, 0),
            BLUE: (0, 0, 255),
            GREEN: (0, 255, 0),
            RED: (255, 0, 0),
            PATH_MARKER: GRAY
        }

    def load_path_file(self, file_path):
        """Lädt eine Pfad-Datei und aktualisiert das Board."""
        try:
            loaded_board = np.load(file_path)
            if loaded_board.shape == (16, 16):
                # Erstelle eine Kopie des Original-Boards
                original_board = np.load(BOARD_FILE)
                # Setze nur die Pfadmarker (6) auf dem Original-Board
                self.board_state = np.where(loaded_board == PATH_MARKER, PATH_MARKER, original_board)
                return True
            else:
                print(f"Warnung: Pfad-Board hat falsche Dimensionen ({loaded_board.shape}).")
                return False
        except Exception as e:
            print(f"Fehler beim Laden von {file_path}: {e}")
            return False

    def next_path(self):
        """Wechselt zum nächsten Pfad in der Liste."""
        if not self.path_files:
            return
        
        self.current_path_index = (self.current_path_index + 1) % len(self.path_files)
        episode_num, file_path = self.path_files[self.current_path_index]
        
        if self.load_path_file(file_path):
            print(f"Geladen: Episode {episode_num}")

    def draw_board(self):
        # Fülle den Hintergrund
        self.screen.fill((200, 200, 200))

        # Zeichne das Brett
        for i in range(self.BOARD_ROWS):
            for j in range(self.BOARD_COLS):
                x1 = self.MARGIN + j * self.FIELD_SIZE
                y1 = self.MARGIN + i * self.FIELD_SIZE

                # Wähle die Farbe basierend auf dem Zustand
                color_val = self.board_state[i][j]
                # Verwende Farbe aus Dict (inkl. Grau für Pfad) oder Standardgrau
                field_color = self.colors.get(color_val, (150, 150, 150))

                pygame.draw.rect(self.screen, field_color, (x1, y1, self.FIELD_SIZE, self.FIELD_SIZE))

                # Zeichne die Linien um das Feld
                pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x1 + self.FIELD_SIZE, y1), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (x1, y1 + self.FIELD_SIZE), (x1 + self.FIELD_SIZE, y1 + self.FIELD_SIZE), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x1, y1 + self.FIELD_SIZE), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (x1 + self.FIELD_SIZE, y1), (x1 + self.FIELD_SIZE, y1 + self.FIELD_SIZE), 1)

                # --- Zeichne Q-Wert-Vektor --- 
                if self.q_table is not None:
                    try:
                        q_vals = self.q_table[i, j]

                        # Softmax Normalisierung für Wahrscheinlichkeiten
                        # Subtrahiere Max für numerische Stabilität
                        stable_q = q_vals - np.max(q_vals)
                        exp_q = np.exp(stable_q)
                        probs = exp_q / np.sum(exp_q)

                        # Handle NaN, falls Summe 0 ist (sollte nicht oft passieren)
                        if np.isnan(probs).any():
                             continue

                        # Aktionen: UP=0, RIGHT=1, DOWN=2, LEFT=3
                        p_up, p_right, p_down, p_left = probs

                        # Berechne Vektorkomponenten
                        dx = p_right - p_left
                        dy = p_down - p_up # Pygame Y ist unten positiv

                        # Zeichne Vektor vom Zentrum
                        center_x = x1 + self.FIELD_SIZE / 2
                        center_y = y1 + self.FIELD_SIZE / 2
                        # Max Länge etwas kleiner als halbes Feld
                        max_len_comp = self.FIELD_SIZE / 2.5 

                        end_x = center_x + dx * max_len_comp
                        end_y = center_y + dy * max_len_comp

                        # Zeichne Pfeillinie
                        pygame.draw.line(self.screen, ARROW_COLOR, (center_x, center_y), (end_x, end_y), ARROW_THICKNESS)

                        # Zeichne Pfeilspitze (nur wenn Vektor nicht Null ist)
                        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                            angle = math.atan2(dy, dx)
                            arrow_len = 5 
                            arrow_angle = math.pi / 6 # 30 Grad

                            # Punkt 1 der Spitze
                            px1 = end_x - arrow_len * math.cos(angle - arrow_angle)
                            py1 = end_y - arrow_len * math.sin(angle - arrow_angle)
                            pygame.draw.line(self.screen, ARROW_COLOR, (end_x, end_y), (px1, py1), ARROW_THICKNESS)

                            # Punkt 2 der Spitze
                            px2 = end_x - arrow_len * math.cos(angle + arrow_angle)
                            py2 = end_y - arrow_len * math.sin(angle + arrow_angle)
                            pygame.draw.line(self.screen, ARROW_COLOR, (end_x, end_y), (px2, py2), ARROW_THICKNESS)

                    except IndexError:
                        # Sollte nicht passieren, wenn Q-Table korrekte Dim hat
                        pass # Ignoriere Fehler für diese Zelle
                    except Exception as e:
                        # Fange andere Fehler ab (z.B. math domain error)
                        pass # Ignoriere Fehler für diese Zelle

        # Zeichne die Buttons
        # Speichern-Button
        pygame.draw.rect(self.screen, (220, 220, 220), self.save_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.save_button_rect, 2)
        font = pygame.font.Font(None, 30)
        text = font.render("Speichern & Beenden", True, (0, 0, 0))
        text_rect = text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text, text_rect)

        # Nächster Pfad-Button
        pygame.draw.rect(self.screen, (220, 220, 220), self.next_path_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.next_path_button_rect, 2)
        if self.path_files:
            episode_num, _ = self.path_files[self.current_path_index]
            text = font.render(f"Nächster Pfad (Episode {episode_num})", True, (0, 0, 0))
        else:
            text = font.render("Keine Pfade verfügbar", True, (0, 0, 0))
        text_rect = text.get_rect(center=self.next_path_button_rect.center)
        self.screen.blit(text, text_rect)

        # Aktualisiere die Anzeige
        pygame.display.flip()

    def handle_click(self, pos):
        # Prüfe, ob der Speichern-Button geklickt wurde
        if self.save_button_rect.collidepoint(pos):
            save_board(self.board_state)
            print("\nBoard-Zustand gespeichert.")
            return True # Signal zum Beenden

        # Prüfe, ob der Nächster-Pfad-Button geklickt wurde
        if self.next_path_button_rect.collidepoint(pos):
            self.next_path()
            return False

        # Berechne die Feldkoordinaten aus den Mauskoordinaten
        x = pos[0] - self.MARGIN
        y = pos[1] - self.MARGIN

        # Prüfe, ob der Klick innerhalb des Bretts war
        if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
            col = x // self.FIELD_SIZE
            row = y // self.FIELD_SIZE

            # Toggle zwischen den Farben (0..4), auch wenn Pfad (6) geklickt wird
            current_color = self.board_state[row][col]
            next_color = (current_color + 1) % NUM_COLORS # Modulo 5 sorgt für 0-4
            self.board_state[row][col] = next_color
            print(f"Feld ({row}, {col}) geändert zu Zustand {next_color}")

        return False # Nicht beenden

    def run(self):
        # Hauptspielschleife
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Fenster geschlossen ohne Speichern.")
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Linke Maustaste
                        if self.handle_click(event.pos):
                            running = False # Beenden nach Button-Klick

            self.draw_board()

        # Beende Pygame
        pygame.quit()

def main():
    # Erstelle und starte den Editor/Viewer
    editor_viewer = BoardEditorViewer()
    editor_viewer.run()

if __name__ == "__main__":
    main() 