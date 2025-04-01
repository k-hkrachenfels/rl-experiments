import numpy as np
import os

# --- Konstanten ---
# Farben (aus board.py abgeleitet)
WHITE = 0
BLACK = 1
BLUE = 2
GREEN = 3
RED = 4

# Aktionen (Indizes)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_MAP = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)} # (d_row, d_col)
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"} # For printing

# Belohnungen
REWARD_DEFAULT = -10
REWARD_BLUE = -100000
# REWARD_GOAL = -10 # Ziel (Grün) hat denselben Reward wie Weiß

# Grid Größe (aus board.py abgeleitet)
GRID_ROWS = 16
GRID_COLS = 16

# --- Lade das Board ---
BOARD_FILE = 'board_save.npy'

def load_environment():
    """Lädt das Board und findet Start/Ziel."""
    if not os.path.exists(BOARD_FILE):
        raise FileNotFoundError(f"Board-Datei '{BOARD_FILE}' nicht gefunden. Bitte erstelle sie (z.B. mit board.py).")
    board = np.load(BOARD_FILE)
    if board.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"Board-Dimensionen in '{BOARD_FILE}' ({board.shape}) entsprechen nicht den erwarteten ({GRID_ROWS}, {GRID_COLS}).")

    green_cells = np.argwhere(board == GREEN)
    if len(green_cells) != 1:
        raise ValueError("Es muss genau ein grünes Start-Feld auf dem Board gefunden werden.")
    
    red_cells = np.argwhere(board == RED)
    if len(red_cells) != 1:
        raise ValueError("Es muss genau ein rotes Ziel-Feld auf dem Board gefunden werden.")


    start_pos = tuple(green_cells[0])
    goal_pos = tuple(red_cells[0])

    print(f"Board geladen ({GRID_ROWS}x{GRID_COLS}).")
    print(f"Startposition (Grün): {start_pos}")
    print(f"Zielposition (Rot): {goal_pos}")
    return board, start_pos, goal_pos

# --- Umgebungsinteraktion ---
def step(state, action, board, goal_pos):
    """
    Führt eine Aktion im Zustand 'state' aus.

    Args:
        state (tuple): Aktuelle Position (row, col).
        action (int): Auszuführende Aktion (UP, DOWN, LEFT, RIGHT).
        board (np.ndarray): Das Spielfeld.
        goal_pos (tuple): Die Zielposition.

    Returns:
        tuple: (next_state, reward, moved)
               next_state (tuple): Die Position nach dem Zug (kann dieselbe sein).
               reward (int): Die erhaltene Belohnung.
               moved (bool): Ob der Agent sich tatsächlich bewegt hat.
               is_done (bool): Ob das Ziel erreicht wurde.
    """
    row, col = state
    d_row, d_col = ACTION_MAP[action]
    next_row, next_col = row + d_row, col + d_col

    # Prüfe, ob die nächste Position im Grid liegt
    if not (0 <= next_row < GRID_ROWS and 0 <= next_col < GRID_COLS):
        # Außerhalb des Grids -> Bleibe stehen, Standard-Reward
        return state, REWARD_DEFAULT, False, False

    target_cell_color = board[next_row][next_col]
    reward = REWARD_DEFAULT
    moved = False
    is_done = False

    if target_cell_color == BLUE:
        reward = REWARD_BLUE
        next_state = state # Bleibe stehen
        moved = False
    elif target_cell_color == WHITE or target_cell_color == GREEN or target_cell_color == RED:
        reward = REWARD_DEFAULT
        next_state = (next_row, next_col) # Gehe zum Feld
        moved = True
        if next_state == goal_pos:
            is_done = True # Ziel erreicht
    elif target_cell_color == BLACK:
        reward = REWARD_DEFAULT
        next_state = state # Bleibe stehen
        moved = False
    else:
        raise ValueError(f"Ungültige Zellfarbe: {target_cell_color}")

    return next_state, reward, moved, is_done

# --- SARSA Agent ---
class SarsaAgent:
    def __init__(self, states_shape, n_actions, alpha=0.05, gamma=0.99, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.3):
        self.n_actions = n_actions
        # Q-Tabelle: Dimensionen des Boards + Anzahl Aktionen
        self.q_table = np.zeros(states_shape + (n_actions,))
        self.alpha = alpha       # Lernrate
        self.gamma = gamma       # Diskontierungsfaktor
        self.epsilon = epsilon     # Explorationsrate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        """Wählt eine Aktion mittels Epsilon-Greedy-Strategie."""
        row, col = state
        # Wähle beste bekannte Aktion
        return np.argmax(self.q_table[row, col])
        
    def choose_action_e_greedy(self, state):
        """Wählt eine Aktion mittels Epsilon-Greedy-Strategie."""
        row, col = state
        if np.random.rand() < self.epsilon:
            # Exploration: Wähle zufällige Aktion
            return np.random.choice(self.n_actions)
        else:
            # Exploitation: Wähle beste bekannte Aktion
            return np.argmax(self.q_table[row, col])

    def update_q_table(self, state, action, reward, next_state, next_action):
        """Aktualisiert die Q-Tabelle gemäß der SARSA-Update-Regel."""
        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[row, col, action]
        next_q = self.q_table[next_row, next_col, next_action]
        target_q = reward + self.gamma * next_q
        new_q = current_q + self.alpha * (target_q - current_q)
        self.q_table[row, col, action] = new_q

    def decay_epsilon(self):
        """Reduziert Epsilon, aber nicht unter das Minimum."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Haupt-Trainingslogik ---
def train():
    try:
        board, start_pos, goal_pos = load_environment()
    except (FileNotFoundError, ValueError) as e:
        print(f"Fehler beim Initialisieren der Umgebung: {e}")
        return

    agent = SarsaAgent(states_shape=board.shape, n_actions=len(ACTIONS))

    num_episodes = 1750 # Anzahl der Trainings-Episoden
    max_steps_per_episode = 1000 # Max Schritte pro Episode, um Endlosschleifen zu vermeiden

    print(f"\nStarte Training über {num_episodes} Episoden (alpha={agent.alpha}, decay={agent.epsilon_decay})...")

    for episode in range(num_episodes):
        state = start_pos
        action = agent.choose_action_e_greedy(state)
        total_reward = 0
        steps = 0

        while steps < max_steps_per_episode:
            steps += 1
            next_state, reward, moved, is_done = step(state, action, board, goal_pos)
            next_action = agent.choose_action(next_state) # Wichtig: Nächste Aktion für SARSA wählen

            # Update Q-Tabelle
            agent.update_q_table(state, action, reward, next_state, next_action)

            total_reward += reward
            state = next_state
            action = agent.choose_action_e_greedy(next_state) # Wichtig: Update für nächsten SARSA-Schritt mit epsilon-greedy

            if is_done:
                break # Ziel erreicht, Episode beendet

        agent.decay_epsilon() # Epsilon nach jeder Episode reduzieren

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} abgeschlossen. Schritte: {steps}, Gesamt-Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    print("\nTraining abgeschlossen.")

    # Optional: Q-Tabelle speichern oder Pfad anzeigen
    # np.save('q_table_sarsa.npy', agent.q_table)
    # print("Q-Tabelle gespeichert in q_table_sarsa.npy")

    # Zeige den gelernten Pfad (optional)
    print("\nDetaillierter Beispielpfad vom Start zum Ziel (State -> Action -> Reward -> NextState):")
    state = start_pos
    visualization_path = [state] # Path for saving to board_with_path.npy
    detailed_path_steps = []     # List to store (state, action, reward, next_state) tuples
    steps = 0
    total_path_reward = 0

    while state != goal_pos and steps < max_steps_per_episode:
        current_state = state
        # Beste Aktion wählen (kein Epsilon)
        action = np.argmax(agent.q_table[current_state[0], current_state[1]])
        # Nächsten Zustand und Reward holen
        next_state, reward, _, is_done = step(current_state, action, board, goal_pos)

        # Speichere den Schritt für die detaillierte Ausgabe
        detailed_path_steps.append((current_state, action, reward, next_state))
        # Speichere nur den Zustand für die Visualisierung
        visualization_path.append(next_state)

        total_path_reward += reward
        state = next_state
        steps += 1

        if is_done:
             print(f"    -> Ziel nach {steps} Schritten erreicht.")
             break
        elif steps >= max_steps_per_episode:
            print("    -> Maximal Schritte erreicht, Ziel nicht gefunden.")
            # Breche die Schleife ab, aber behalte die bisherigen Schritte
            break

    # Gib den detaillierten Pfad aus
    for i, (s, a, r, ns) in enumerate(detailed_path_steps):
        print(f"    Schritt {i+1}: Zustand={s}, Aktion={ACTION_NAMES[a]}, Reward={r}, Nächster Zustand={ns}")

    print(f"    Gesamt-Reward des Pfades: {total_path_reward}")
    # print(f"    Pfad (nur Zustände): {visualization_path}") # Optional: Nur Zustandssequenz anzeigen

    # Füge die neue Zusammenfassungszeile hinzu
    print(f"\nZusammenfassung: Optimaler Pfad hat {len(detailed_path_steps)} Schritte mit Gesamt-Reward {total_path_reward}.")

    # Erstelle und speichere das Board mit dem Pfad (verwendet visualization_path)
    board_with_path = board.copy()
    PATH_MARKER = 6 # Neue Konstante für den Pfad
    # Verwende visualization_path für die Markierung auf dem Board
    for r, c in visualization_path:
        # Markiere Pfad nur auf erlaubten Feldern (Weiß, Grün), nicht Start/Ziel selbst
        if (r, c) != start_pos and (r, c) != goal_pos and board_with_path[r, c] in [WHITE, GREEN]:
            board_with_path[r, c] = PATH_MARKER

    np.save('board_with_path.npy', board_with_path)
    print(f"Board mit Pfad gespeichert in 'board_with_path.npy'")


if __name__ == "__main__":
    train() 