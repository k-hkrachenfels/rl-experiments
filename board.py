import numpy as np
import pygame
import math
from config import Config, load_config, Color
from persistence import load_editor_viewer_data, save_board, load_path_file


ARROW_COLOR = (100, 0, 0) # Dark red for arrows
ARROW_THICKNESS = 2

class BoardEditorViewer:
    def __init__(self, config):
        self.config = config
        self.FIELD_SIZE = 30
        self.BOARD_ROWS = 16
        self.BOARD_COLS = 16
        self.BOARD_WIDTH = self.FIELD_SIZE * self.BOARD_COLS
        self.BOARD_HEIGHT = self.FIELD_SIZE * self.BOARD_ROWS
        self.MARGIN = 20
        self.BUTTON_HEIGHT = 40
        self.WINDOW_WIDTH = self.BOARD_WIDTH + 2*self.MARGIN
        self.WINDOW_HEIGHT = self.BOARD_HEIGHT + 6*self.MARGIN + 4*self.BUTTON_HEIGHT  # Space for board + margins + 4 buttons

        # Initialize Pygame
        pygame.init()

        # Create the window
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Board Editor/Viewer with Q-Values (16x16)")

        # Setup mode flag (True = setup mode, False = view mode)
        self.setup_mode = False
        # Best action mode flag (True = show only best action, False = show weighted vector)
        self.show_best_action = False

        # Load initial state (prioritizes path)
        self.board_state, self.path_initially_loaded, self.q_table, self.path_files = load_editor_viewer_data(config)
        self.current_path_index = min(self.path_files.keys()) if self.path_files else -1
        
        if self.q_table is not None:
            print("Display: Q-values are being visualized.")
        else:
            print("Display: No Q-values found/loaded.")

        # Create the buttons
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
        
        self.setup_button_rect = pygame.Rect(
            self.MARGIN,
            self.BOARD_HEIGHT + 4*self.MARGIN + 2*self.BUTTON_HEIGHT,
            self.BOARD_WIDTH,
            self.BUTTON_HEIGHT
        )

        self.best_action_button_rect = pygame.Rect(
            self.MARGIN,
            self.BOARD_HEIGHT + 5*self.MARGIN + 3*self.BUTTON_HEIGHT,
            self.BOARD_WIDTH,
            self.BUTTON_HEIGHT
        )

        # Color mapping for the display (incl. path)
        self.colors = {
            Color.WHITE: (255, 255, 255),
            Color.BLACK: (0, 0, 0),
            Color.BLUE: (0, 0, 255),
            Color.GREEN: (0, 255, 0),
            Color.RED: (255, 0, 0),
            Color.GRAY: (128 ,128, 128)
        }

    def next_path(self):
        """Switches to the next path in the list."""
        if not self.path_files:
            return
        
        # Convert dictionary keys to a list and sort them
        episode_numbers = sorted(self.path_files.keys())
        if not episode_numbers:
            return
        
        # Find the current index
        current_index = episode_numbers.index(self.current_path_index)
        # Go to the next index (or back to the beginning)
        next_index = (current_index + 1) % len(episode_numbers)
        self.current_path_index = episode_numbers[next_index]
        
        # Load the new episode
        board_file, q_file = self.path_files[self.current_path_index]
        self.board_state = load_path_file(self.config, board_file)
        if self.board_state is not None:
            print(f"Loaded: Episode {self.current_path_index}")
            
            # Also load the corresponding Q-table
            try:
                self.q_table = np.load(q_file)
                print(f"Q-table for episode {self.current_path_index} loaded")
            except Exception as e:
                print(f"Error loading the Q-table: {e}")

    def draw_board(self):
        # Fill the background
        self.screen.fill((200, 200, 200))

        # Draw the board
        for i in range(self.BOARD_ROWS):
            for j in range(self.BOARD_COLS):
                x1 = self.MARGIN + j * self.FIELD_SIZE
                y1 = self.MARGIN + i * self.FIELD_SIZE

                # Choose the color based on the state
                color_val = self.board_state[i][j]
                # Use color from dict (incl. gray for path) or default = lighter gray
                field_color = self.colors.get(color_val, (150, 150, 150))

                pygame.draw.rect(self.screen, field_color, (x1, y1, self.FIELD_SIZE, self.FIELD_SIZE))

                # Draw the lines around the field
                pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x1 + self.FIELD_SIZE, y1), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (x1, y1 + self.FIELD_SIZE), (x1 + self.FIELD_SIZE, y1 + self.FIELD_SIZE), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x1, y1 + self.FIELD_SIZE), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (x1 + self.FIELD_SIZE, y1), (x1 + self.FIELD_SIZE, y1 + self.FIELD_SIZE), 1)

                # --- Draw Q-value vector (only in view mode) ---
                if self.q_table is not None and not self.setup_mode:
                    try:
                        q_vals = self.q_table[i, j]

                        if self.show_best_action:
                            # --- Draw single arrow for best action (Greedy Policy) ---
                            # Find index of max Q-value
                            best_action_idx = np.argmax(q_vals)
                            
                            # Map index to direction (UP=0, RIGHT=1, DOWN=2, LEFT=3)
                            # Directions: (dx, dy) where dy is positive downwards
                            directions = {
                                0: (0, -1),  # UP
                                1: (1, 0),   # RIGHT
                                2: (0, 1),   # DOWN
                                3: (-1, 0)   # LEFT
                            }
                            
                            dx, dy = directions.get(best_action_idx, (0, 0))
                            
                            # Draw vector from the center
                            center_x = x1 + self.FIELD_SIZE / 2
                            center_y = y1 + self.FIELD_SIZE / 2
                            
                            # Length for best action arrow
                            arrow_len_pixels = self.FIELD_SIZE / 2.2
                            
                            end_x = center_x + dx * arrow_len_pixels
                            end_y = center_y + dy * arrow_len_pixels
                            
                            # Use a slightly different color for best action to distinguish
                            BEST_ARROW_COLOR = (0, 0, 150) # Dark blue
                            
                            # Draw arrow line
                            pygame.draw.line(self.screen, BEST_ARROW_COLOR, (center_x, center_y), (end_x, end_y), ARROW_THICKNESS + 1)
                            
                            # Draw arrowhead
                            angle = math.atan2(dy, dx)
                            tip_len = 7
                            tip_angle = math.pi / 6
                            
                            px1 = end_x - tip_len * math.cos(angle - tip_angle)
                            py1 = end_y - tip_len * math.sin(angle - tip_angle)
                            pygame.draw.line(self.screen, BEST_ARROW_COLOR, (end_x, end_y), (px1, py1), ARROW_THICKNESS + 1)
                            
                            px2 = end_x - tip_len * math.cos(angle + tip_angle)
                            py2 = end_y - tip_len * math.sin(angle + tip_angle)
                            pygame.draw.line(self.screen, BEST_ARROW_COLOR, (end_x, end_y), (px2, py2), ARROW_THICKNESS + 1)

                        else:
                            # --- Draw weighted vector (Stochastic/Exploration view) ---
                            # Softmax normalization for probabilities
                            # Subtract max for numerical stability
                            stable_q = q_vals - np.max(q_vals)
                            exp_q = np.exp(stable_q)
                            probs = exp_q / np.sum(exp_q)
    
                            # Handle NaN if sum is 0 (should not happen often)
                            if np.isnan(probs).any():
                                 continue
    
                            # Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
                            p_up, p_right, p_down, p_left = probs
    
                            # Calculate vector components
                            dx = p_right - p_left
                            dy = p_down - p_up # Pygame Y is positive downwards
    
                            # Draw vector from the center
                            center_x = x1 + self.FIELD_SIZE / 2
                            center_y = y1 + self.FIELD_SIZE / 2
                            # Max length slightly smaller than half a field
                            max_len_comp = self.FIELD_SIZE / 2.5 
    
                            end_x = center_x + dx * max_len_comp
                            end_y = center_y + dy * max_len_comp
    
                            # Draw arrow line
                            pygame.draw.line(self.screen, ARROW_COLOR, (center_x, center_y), (end_x, end_y), ARROW_THICKNESS)
    
                            # Draw arrowhead (only if vector is not zero)
                            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                                angle = math.atan2(dy, dx)
                                arrow_len = 5 
                                arrow_angle = math.pi / 6 # 30 degrees
    
                                # Point 1 of the tip
                                px1 = end_x - arrow_len * math.cos(angle - arrow_angle)
                                py1 = end_y - arrow_len * math.sin(angle - arrow_angle)
                                pygame.draw.line(self.screen, ARROW_COLOR, (end_x, end_y), (px1, py1), ARROW_THICKNESS)
    
                                # Point 2 of the tip
                                px2 = end_x - arrow_len * math.cos(angle + arrow_angle)
                                py2 = end_y - arrow_len * math.sin(angle + arrow_angle)
                                pygame.draw.line(self.screen, ARROW_COLOR, (end_x, end_y), (px2, py2), ARROW_THICKNESS)

                    except IndexError:
                        # Should not happen if Q-table has correct dimensions
                        pass # Ignore errors for this cell
                    except Exception as e:
                        # Catch other errors (e.g. math domain error)
                        pass # Ignore errors for this cell

        # Draw the buttons
        # Save/Exit button
        pygame.draw.rect(self.screen, (220, 220, 220), self.save_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.save_button_rect, 2)
        font = pygame.font.Font(None, 30)
        
        button_label = "Save & Exit" if self.setup_mode else "Exit"
        
        text = font.render(button_label, True, (0, 0, 0))
        text_rect = text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text, text_rect)

        # Next path button (only visible in view mode)
        if not self.setup_mode:
            pygame.draw.rect(self.screen, (220, 220, 220), self.next_path_button_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.next_path_button_rect, 2)
            if self.path_files:
                current_episode = self.current_path_index
                text = font.render(f"Episode {current_episode} - Next Path", True, (0, 0, 0))
            else:
                text = font.render("No paths available", True, (0, 0, 0))
            text_rect = text.get_rect(center=self.next_path_button_rect.center)
            self.screen.blit(text, text_rect)
        
        # Setup/View mode toggle button
        button_color = (100, 200, 100) if self.setup_mode else (200, 200, 100)
        pygame.draw.rect(self.screen, button_color, self.setup_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.setup_button_rect, 2)
        mode_text = "Switch to View Mode" if self.setup_mode else "Switch to Setup Mode"
        text = font.render(mode_text, True, (0, 0, 0))
        text_rect = text.get_rect(center=self.setup_button_rect.center)
        self.screen.blit(text, text_rect)

        # Best Action / Weighted toggle button (only in View Mode)
        if not self.setup_mode:
            ba_color = (150, 150, 250) if self.show_best_action else (200, 200, 220)
            pygame.draw.rect(self.screen, ba_color, self.best_action_button_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.best_action_button_rect, 2)
            ba_text = "Show Weighted Q-Values" if self.show_best_action else "Show Winning Path"
            text = font.render(ba_text, True, (0, 0, 0))
            text_rect = text.get_rect(center=self.best_action_button_rect.center)
            self.screen.blit(text, text_rect)

        # Update the display
        pygame.display.flip()

    def handle_click(self, pos):
        # Check if the save/exit button was clicked
        if self.save_button_rect.collidepoint(pos):
            if self.setup_mode:
                save_board(self.config, self.board_state)
                print("\nBoard state saved.")
            else:
                print("\nExiting without saving (View Mode).")
            return True # Signal to exit

        # Check if the next path button was clicked (only in view mode)
        if not self.setup_mode and self.next_path_button_rect.collidepoint(pos):
            self.next_path()
            return False
        
        # Check if the setup mode toggle button was clicked
        if self.setup_button_rect.collidepoint(pos):
            self.setup_mode = not self.setup_mode
            mode_name = "Setup" if self.setup_mode else "View"
            print(f"Switched to {mode_name} mode")
            
            if self.setup_mode:
                # Clear path fields (Gray/Visited >= 5) to make them editable structural layout again
                # Assuming board_state is a numpy array
                self.board_state[self.board_state >= 5] = 0
                print("Cleared path visualizations (grey fields) for setup.")
                
            return False
        
        # Check if Best Action button was clicked (only in View Mode)
        if not self.setup_mode and self.best_action_button_rect.collidepoint(pos):
            self.show_best_action = not self.show_best_action
            print(f"Visualization mode: {'Best Action' if self.show_best_action else 'Weighted Q-Values'}")
            return False

        # Calculate the field coordinates from the mouse coordinates
        x = pos[0] - self.MARGIN
        y = pos[1] - self.MARGIN

        # Check if the click was inside the board
        if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
            col = x // self.FIELD_SIZE
            row = y // self.FIELD_SIZE

            # Toggle between colors (0..4), even if path (6) is clicked
            current_color = self.board_state[row][col]
            next_color = (current_color + 1) % (len(Color)-1) # All but one colors are toggable (this assumes that the last color is used as marker for the path and is not available to the editor)
            self.board_state[row][col] = next_color
            print(f"Field ({row}, {col}) changed to state {next_color}")

        return False # Do not exit

    def run(self):
        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Window closed without saving.")
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if self.handle_click(event.pos):
                            running = False # Exit after button click

            self.draw_board()

        # Quit Pygame
        pygame.quit()


def main():
    config = load_config()
    editor_viewer = BoardEditorViewer(config)
    editor_viewer.run()

if __name__ == "__main__":
    main()