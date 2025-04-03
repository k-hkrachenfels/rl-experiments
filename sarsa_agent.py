import numpy as np
import os
import glob

# --- Konstanten ---
# Farben (aus board.py abgeleitet)
WHITE = 0
BLACK = 1
BLUE = 2
GREEN = 3
RED = 4

# Aktionen (Indizes)
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
# Aktionen als Liste
ACTIONS = [UP, RIGHT, DOWN, LEFT] # Indices 0, 1, 2, 3
# Mapping von Aktion zu (delta_row, delta_col)
ACTION_MAP = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)} # (d_row, d_col)
# Namen der Aktionen für die Ausgabe
ACTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"} # For printing

# Belohnungen
REWARD_DEFAULT = -10
REWARD_BLUE = -100000
# REWARD_GOAL = -10 # Ziel (Grün) hat denselben Reward wie Weiß

# Grid Größe (aus board.py abgeleitet)
GRID_ROWS = 16
GRID_COLS = 16

# --- Lade das Board ---
BOARD_FILE = 'board_save.npy'

# ANSI color codes für Konsolenausgabe
ANSI_RED = '\033[91m'
ANSI_GREEN = '\033[92m'
ANSI_YELLOW = '\033[93m'
ANSI_BLUE = '\033[94m'
ANSI_RESET = '\033[0m'  # Reset color

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
    # Checkt ob action gültig ist bevor es benutzt wird
    if action not in ACTION_MAP:
        raise ValueError(f"Unbekannte Aktion: {action}")
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
    def __init__(self, states_shape, n_actions, alpha=0.05, gamma=0.99, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.01):
        self.n_actions = n_actions
        # Q-Tabelle mit Dimensionen (rows, cols, num_actions)
        self.q_table = np.zeros(states_shape + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        """Wählt die beste Aktion basierend auf den Q-Werten (gierig)."""
        row, col = state
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
        """Reduziert Epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def delete_old_episode_files():
    """Löscht alle alten Board-Dateien mit Pfad aus vorherigen Episoden."""
    files = glob.glob('board_with_path_*.npy')
    for file in files:
        try:
            os.remove(file)
            print(f"Gelöscht: {file}")
        except Exception as e:
            print(f"Fehler beim Löschen von {file}: {e}")

def save_results(board, agent, visualization_path, detailed_path_steps, start_pos, goal_pos, output_file='board_with_path.npy'):
    """
    Speichert die Trainingsergebnisse in Dateien.
    
    Args:
        board (np.ndarray): Das Spielfeld
        agent (SarsaAgent): Der trainierte Agent
        visualization_path (list): Liste der Zustände für die Visualisierung
        detailed_path_steps (list): Liste der detaillierten Schritte (state, action, reward, next_state)
        start_pos (tuple): Startposition
        goal_pos (tuple): Zielposition
        output_file (str): Name der Ausgabedatei für das Board mit Pfad
    """
    # Erstelle und speichere das Board mit dem Pfad
    board_with_path = board.copy()
    PATH_MARKER = 6  # Neue Konstante für den Pfad
    # Verwende visualization_path für die Markierung auf dem Board
    for r, c in visualization_path:
        # Markiere Pfad nur auf erlaubten Feldern (Weiß, Grün), nicht Start/Ziel selbst
        if (r, c) != start_pos and (r, c) != goal_pos and board_with_path[r, c] in [WHITE, GREEN]:
            board_with_path[r, c] = PATH_MARKER

    np.save(output_file, board_with_path)

    # Speichere die Q-Tabelle
    try:
        q_table_to_save = agent.q_table

        # Überprüfe die Dimensionen (sollte 16, 16, 4 sein)
        if q_table_to_save.shape != (GRID_ROWS, GRID_COLS, len(ACTIONS)):
            print(f"Warnung: Unerwartete Q-Tabellen-Dimension beim Speichern: {q_table_to_save.shape}")

        # Speichere die Tabelle direkt
        np.save('q_table_final.npy', q_table_to_save)
    except Exception as e:
        print(f"Fehler beim Speichern der Q-Tabelle: {e}")

def print_path_details(episode_number, detailed_path_steps, total_path_reward):
    """
    Gibt die Details eines Pfades aus.
    
    Args:
        episode_number (int): Nummer der Episode
        detailed_path_steps (list or int): Liste der Schritte oder Anzahl der Schritte
        total_path_reward (float): Gesamtreward des Pfades
    """
    if isinstance(detailed_path_steps, list):
        steps = len(detailed_path_steps)
    else:
        steps = detailed_path_steps
    print(f"{ANSI_RED}Val   {episode_number}, {steps} steps, Reward {total_path_reward}.{ANSI_RESET}")

def find_optimal_path(agent, board, start_pos, goal_pos, max_steps_per_episode):
    """
    Findet den optimalen Pfad vom Start zum Ziel basierend auf den gelernten Q-Werten.
    
    Args:
        agent (SarsaAgent): Der trainierte Agent
        board (np.ndarray): Das Spielfeld
        start_pos (tuple): Startposition
        goal_pos (tuple): Zielposition
        max_steps_per_episode (int): Maximale Anzahl von Schritten pro Episode
        
    Returns:
        tuple: (visualization_path, detailed_path_steps, total_path_reward)
            - visualization_path: Liste der Zustände für die Visualisierung
            - detailed_path_steps: Liste der detaillierten Schritte (state, action, reward, next_state)
            - total_path_reward: Gesamter Reward des Pfades
    """
    state = start_pos
    visualization_path = [state]  # Path for saving to board_with_path.npy
    detailed_path_steps = []      # List to store (state, action, reward, next_state) tuples
    steps = 0
    total_path_reward = 0

    while state != goal_pos and steps < max_steps_per_episode:
        current_state = state
        # Beste Aktion wählen (gierig)
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

    return visualization_path, detailed_path_steps, total_path_reward

def validate(agent, board, start_pos, goal_pos, max_steps_per_episode, episode_number):
    """
    Führt die Validierung des trainierten Agenten durch:
    1. Findet den optimalen Pfad
    2. Zeigt die Details des Pfades an
    3. Speichert die Ergebnisse
    
    Args:
        agent (SarsaAgent): Der trainierte Agent
        board (np.ndarray): Das Spielfeld
        start_pos (tuple): Startposition
        goal_pos (tuple): Zielposition
        max_steps_per_episode (int): Maximale Anzahl von Schritten pro Episode
        episode_number (int, optional): Nummer der aktuellen Episode für die Dateibenennung
    """
    
    # 1. Finde den optimalen Pfad
    visualization_path, detailed_path_steps, total_path_reward = find_optimal_path(
        agent, board, start_pos, goal_pos, max_steps_per_episode
    )

    # 2. Zeige die Details des Pfades an
    print_path_details(episode_number, detailed_path_steps, total_path_reward)

    # 3. Speichere die Ergebnisse
    if episode_number is not None:
        output_file = f'board_with_path_{episode_number+1:06d}.npy'
    else:
        output_file = 'board_with_path.npy'
    
    # Speichere die Q-Tabelle und das Board mit dem Pfad
    save_results(board, agent, visualization_path, detailed_path_steps, start_pos, goal_pos, output_file)
    

def train(num_episodes=4000, max_steps_per_episode=500, validate_interval=100):
    """
    Trainiert den SARSA-Agenten.
    
    Args:
        num_episodes (int): Anzahl der Trainings-Episoden (Standard: 1750)
        max_steps_per_episode (int): Maximale Schritte pro Episode (Standard: 500)
        validate_interval (int): Intervall für Validierung in Episoden (Standard: 100)
    """
    try:
        # Ensure environment loads a 16x16 board
        board, start_pos, goal_pos = load_environment()
        if board.shape != (GRID_ROWS, GRID_COLS):
             raise ValueError(f"Loaded board shape {board.shape} does not match GRID_ROWS/COLS ({GRID_ROWS},{GRID_COLS})")
    except (FileNotFoundError, ValueError) as e:
        print(f"Fehler beim Initialisieren der Umgebung: {e}")
        return

    # Lösche alte Episode-Dateien
    print("\nLösche alte Episode-Dateien...")
    delete_old_episode_files()

    agent = SarsaAgent(states_shape=board.shape, n_actions=len(ACTIONS))

    print(f"\nStarte Training über {num_episodes} Episoden (alpha={agent.alpha}, decay={agent.epsilon_decay}) für {GRID_ROWS}x{GRID_COLS} Grid...")

    for episode in range(num_episodes):
        state = start_pos
        action = agent.choose_action_e_greedy(state)
        total_reward = 0
        steps = 0
        if episode % validate_interval == 0:
            # Führe Validierung vor jeder validate_interval-ten Episode durch, die erste Validierung erfolgt mit den initialen Q-Werten
            validate(agent, board, start_pos, goal_pos, max_steps_per_episode, episode)
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

        if (episode ) % validate_interval == 0:
            print(f"Train {episode}, {steps} steps, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    print("\nTraining abgeschlossen.")

    # Führe die finale Validierung durch
    validate(agent, board, start_pos, goal_pos, max_steps_per_episode, num_episodes)

if __name__ == "__main__":
    train() 