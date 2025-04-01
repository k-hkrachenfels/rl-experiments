import numpy as np
import pygame
import sys
import os
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

# --- Board Laden/Speichern (vereinfacht, da sarsa_agent die Pfad-Datei erstellt) ---
BOARD_FILE_ORIGINAL = 'board_save.npy' # Original vom Editor
BOARD_FILE_WITH_PATH = 'board_with_path.npy' # Vom Agenten generiert
BOARD_FILE = 'board_save.npy'

def load_editor_viewer_board():
    """
    Lädt Board für Editor/Viewer (16x16).
    Prioritisiert Pfad-Datei für initialen View, dann Editor-Datei, dann neu.
    Returns: tuple (board_state: np.ndarray, path_was_loaded: bool)
    """
    path_loaded = False
    # 1. Versuche Pfad-Datei zu laden
    if os.path.exists(BOARD_FILE_WITH_PATH):
        print(f"Versuche, Pfad-Board zu laden: {BOARD_FILE_WITH_PATH}")
        try:
            board = np.load(BOARD_FILE_WITH_PATH)
            if board.shape == (16, 16):
                print("Pfad-Board geladen.")
                return board, True # Pfad-Board zurückgeben
            else:
                print(f"Warnung: Pfad-Board hat falsche Dimensionen ({board.shape}). Ignoriere.")
        except Exception as e:
            print(f"Fehler beim Laden von {BOARD_FILE_WITH_PATH}: {e}. Ignoriere.")

    # 2. Wenn Pfad nicht geladen, versuche Editor-Datei
    if os.path.exists(BOARD_FILE):
        print(f"Lade Editor-Board: {BOARD_FILE}")
        try:
            board = np.load(BOARD_FILE)
            if board.shape == (16, 16):
                print("Editor-Board geladen.")
                return board, False # Editor-Board zurückgeben
            else:
                print(f"Warnung: Editor-Board hat falsche Dimensionen ({board.shape}). Erstelle neues Board.")
        except Exception as e:
            print(f"Fehler beim Laden von {BOARD_FILE}: {e}. Erstelle neues Board.")

    # 3. Wenn beides fehlschlägt, erstelle neues Board
    print("Keine gültige Board-Datei gefunden. Erstelle neues leeres 16x16 Board.")
    return np.zeros((16, 16), dtype=int), False # Neues Board zurückgeben

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

class BoardEditorViewer: # Umbenannt
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
        self.WINDOW_HEIGHT = self.BOARD_HEIGHT + 2*self.MARGIN + self.BUTTON_HEIGHT

        # Initialisiere Pygame
        pygame.init()

        # Erstelle das Fenster
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Board Editor/Viewer (16x16)") # Titel angepasst

        # Lade initialen Zustand (priorisiert Pfad)
        self.board_state, self.path_initially_loaded = load_editor_viewer_board()
        if self.path_initially_loaded:
             print("Anzeige: Pfad-Board initial geladen.")
        else:
             print("Anzeige: Editor-Board oder neues Board geladen.")

        # Erstelle den Button
        self.button_rect = pygame.Rect(
            self.MARGIN,
            self.BOARD_HEIGHT + 2*self.MARGIN,
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
            PATH_MARKER: GRAY # Pfadfarbe wieder hinzugefügt
        }

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

        # Zeichne den Button
        pygame.draw.rect(self.screen, (220, 220, 220), self.button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.button_rect, 2)

        # Text für den Button
        font = pygame.font.Font(None, 30)
        text = font.render("Speichern & Beenden", True, (0, 0, 0))
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)

        # Aktualisiere die Anzeige
        pygame.display.flip()

    def handle_click(self, pos):
        # Prüfe, ob der Button geklickt wurde
        if self.button_rect.collidepoint(pos):
            save_board(self.board_state) # Speichert IMMER in board_save.npy
            print("\nBoard-Zustand gespeichert.")
            # print(self.board_state) # Optional: Matrix anzeigen
            return True # Signal zum Beenden

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
            # Optional: Flag setzen, dass Bearbeitung stattfand, falls benötigt
            # self.edited = True

        return False # Nicht beenden

    def run(self):
        # Hauptspielschleife
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Hinweis: Schließen ohne Speichern
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