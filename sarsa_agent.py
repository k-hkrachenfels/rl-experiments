import glob
import numpy as np
import os

from abc import ABC, abstractmethod
from config import Config, load_config, AgentType, GridState
from dataclasses import dataclass
from enum import Enum
from persistence import load_board_and_find_start_goal, save_results, delete_old_files
from typing import Tuple, List
from util import IndexableEnumMeta

# an Enum that also supports indexing and applying len (by using specific metaclass)
class Action(Enum, metaclass=IndexableEnumMeta):
    UP    = (-1, 0)
    RIGHT = (0, 1)
    DOWN  = (1, 0)
    LEFT  = (0, -1)   

@dataclass
class Board:
    board_values: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
class BaseAgent(ABC):
    def __init__(self, board, config):
        self.grid_rows = config.world.grid_rows
        self.grid_cols = config.world.grid_cols
        self.n_actions = len(Action)
        self.alpha = config.agent.alpha
        self.gamma = config.agent.gamma
        self.epsilon = config.agent.epsilon
        self.epsilon_decay = config.agent.epsilon_decay
        self.epsilon_min = config.agent.epsilon_min
        self.q_table = np.zeros((self.grid_rows, self.grid_cols, self.n_actions))
        self.board_values = board.board_values #todo: confusing use of variable names
        self.start = board.start
        self.goal = board.goal
        self.env = None

    def choose_action(self, state):
        """Wählt eine Aktion basierend auf der aktuellen Q-Tabelle."""
        return np.argmax(self.q_table[state])

    def choose_action_e_greedy(self, state):
        """Wählt eine Aktion mit Epsilon-Greedy Strategie."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self.choose_action(state)

    def decay_epsilon(self):
        """Reduziert Epsilon nach jeder Episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_optimal_path_based_on_qtable(self, board, start_pos, goal_pos, max_steps_per_episode, config: Config)-> Tuple[List[Tuple[int,int]], int]:
        """
        Validation based on  Q-values
        
        Args:
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
        path = [state]  # Path for saving to board_with_path.npy
        steps = 0
        total_reward = 0
        is_done = False

        while state != goal_pos and steps < max_steps_per_episode and not is_done:
            current_state = state
            # greedy take next action / best from q-table
            action = np.argmax(self.q_table[current_state[0], current_state[1]])
            # Nächsten Zustand und Reward holen
            next_state, reward, _, is_done = self.apply_step(current_state, action, board, goal_pos, config)
            path.append(next_state)
            total_reward += reward
            state = next_state
            steps += 1

        return path, total_reward
    
    def train(self, config: Config):
        """
        Trainiert den Agenten basierend auf der Konfiguration.
        """
        print(f"\nStarte Training über {config.training.num_episodes} Episoden (alpha={config.agent.alpha}, decay={config.agent.epsilon_decay}) für {config.world.grid_rows}x{config.world.grid_cols} Grid...")

        for episode in range(config.training.num_episodes):
            state = self.start
            action = self.choose_action_e_greedy(state)
            total_reward = 0
            steps = 0

            while steps < config.training.max_steps_per_episode:
                steps += 1
                next_state, reward, moved, is_done = self.apply_step(state, action, self.board_values, self.goal, config)
                
                if isinstance(self, QLearningAgent):
                    next_action = self.choose_action_e_greedy(next_state)
                    self.update_q_table(state, action, reward, next_state, next_action)
                else:
                    next_action = self.choose_action(next_state)
                    self.update_q_table(state, action, reward, next_state, next_action)

                total_reward += reward
                state = next_state
                action = self.choose_action_e_greedy(next_state)

                if is_done:
                    break

            self.decay_epsilon()

            if (episode + 1) % config.training.validate_interval == 0:

                self.validate(config, self.board_values, self.start, self.goal, config.training.max_steps_per_episode, episode, self.grid_rows, self.grid_cols)
                print(f"train {episode + 1}, {steps} steps, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        print("\nTraining abgeschlossen.")

        # Führe die finale Validierung durch
        self. validate(config, self.board_values, self.start, self.goal, config.training.max_steps_per_episode, episode, self.grid_rows, self.grid_cols)

    def validate(self, config: Config, 
                board: np.ndarray, 
                start_pos: tuple[int,int],
                goal_pos: tuple[int, int], 
                max_steps_per_episode: int,
                episode_number: int, 
                grid_rows: int, grid_cols:int
                ):
        # 1. Finde den optimalen Pfad
        path, total_reward = self.calculate_optimal_path_based_on_qtable( board, start_pos, goal_pos, max_steps_per_episode, config
        )
        print(f"validation Episode {episode_number+1}, {len(path)} steps, Reward {total_reward}.")
        
        # Speichere die Q-Tabelle und das Board mit dem Pfad
        save_results(config, board, self, path, start_pos, goal_pos, episode_number)  

    def apply_step(self, state, action, board, goal_pos, config: Config):
        """
        Executes an action in the 'state'.
        Blue cells: Path ends here with rewards.invalid from the configuration (blue is water)
        Black cells: Standard reward from the configuration but no movement
        White cells: Standard reward from the configuration and movement
        Green cells: Standard reward from the configuration and target reached (target is green)
        Red cells: Standard reward from the configuration and movement (start field is red)

        Args:
            state (tuple): Current position (row, col).
            action (int): Action to be executed (UP, DOWN, LEFT, RIGHT).
            board (np.ndarray): The game board.
            goal_pos (tuple): The target position.
            config (Config): The agent's configuration.

        Returns:
            tuple: (next_state, reward, moved, is_done)
                next_state (tuple): The position after the move (can be the same).
                reward (int): The reward received.
                moved (bool): Whether the agent actually moved.
                is_done (bool): Whether the target was reached or an invalid state was entered.
        """
        row, col = state
        # checks if action is in correct range
        if action >= len(Action):
            raise ValueError(f"Unknown action: {action}")
        
        d_row, d_col = Action[action].value
        next_row, next_col = row + d_row, col + d_col

        # Check if the next position is within the grid
        if not (0 <= next_row < config.world.grid_rows and 0 <= next_col < config.world.grid_cols):
            # Outside the grid -> Stay put, default reward
            return state, config.rewards.default, False, False

        target_cell_color = board[next_row][next_col]
        reward = config.rewards.default
        moved = False
        is_done = False

        if target_cell_color == GridState.INVALID:
            reward = config.rewards.invalid
            next_state = state # invalid field (water), end of path
            is_done = True
            moved = False
        elif target_cell_color in [GridState.FREE, GridState.START, GridState.TARGET, GridState.VISITED]:
            reward = config.rewards.default
            next_state = (next_row, next_col) # next cell
            moved = True
            if next_state == goal_pos:
                is_done = True # target reached, done
        elif target_cell_color == GridState.WALL:
            reward = config.rewards.default
            next_state = state # stay on current cell but discount with default
            moved = False
        else:
            raise ValueError(f"Invalid cell color: {target_cell_color}")

        return next_state, reward, moved, is_done




    @abstractmethod
    def update_q_table(self, state, action, reward, next_state, next_action):
        """Diese Methode wird von den Unterklassen implementiert."""
        pass

class SarsaAgent(BaseAgent):
    def update_q_table(self, state, action, reward, next_state, next_action):
        """SARSA Update: Q(s,a) = Q(s,a) + α[R + γQ(s',a') - Q(s,a)]"""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

class QLearningAgent(BaseAgent):
    def update_q_table(self, state, action, reward, next_state, next_action):
        """Q-Learning Update: Q(s,a) = Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]"""
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

def agent_init(config: Config):
    """Loads the board/world and start and goal position """
    board, start, goal = load_board_and_find_start_goal(config)
    board_obj = Board(board_values=board, start=start, goal=goal)
    if config.agent.type == AgentType.SARSA:
        agent = SarsaAgent(board_obj, config)
        print("Verwende SARSA Agent")
    elif config.agent.type == AgentType.Q_LEARNING:
        agent = QLearningAgent(board_obj, config)
        print("Verwende Q-Learning Agent")
    else:
        raise ValueError(f"Unbekannter Agent-Typ: {config.agent.type}. ")
    return agent


if __name__ == "__main__":

    print("\nLade Konfiguration...")
    config = load_config()

    print("\nLösche alte Episoden- und Q-Value-Dateien...")
    delete_old_files(config)

    print("\nInitialisiere Agenten...")
    agent = agent_init(config)

    print("\nTrainiere Agenten...")
    if config:
        agent.train(config)
