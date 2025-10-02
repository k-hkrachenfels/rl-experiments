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
        self.WINDOW_HEIGHT = self.BOARD_HEIGHT + 2*self.MARGIN + 2*self.BUTTON_HEIGHT + 10  # Extra height for second button

        # Initialize Pygame
        pygame.init()

        # Create the window
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Board Editor/Viewer with Q-Values (16x16)")

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

                # --- Draw Q-value vector ---
                if self.q_table is not None:
                    try:
                        q_vals = self.q_table[i, j]

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
        # Save button
        pygame.draw.rect(self.screen, (220, 220, 220), self.save_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.save_button_rect, 2)
        font = pygame.font.Font(None, 30)
        text = font.render("Save & Exit", True, (0, 0, 0))
        text_rect = text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text, text_rect)

        # Next path button
        pygame.draw.rect(self.screen, (220, 220, 220), self.next_path_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.next_path_button_rect, 2)
        if self.path_files:
            current_episode = self.current_path_index
            text = font.render(f"Episode {current_episode} - Next Path", True, (0, 0, 0))
        else:
            text = font.render("No paths available", True, (0, 0, 0))
        text_rect = text.get_rect(center=self.next_path_button_rect.center)
        self.screen.blit(text, text_rect)

        # Update the display
        pygame.display.flip()

    def handle_click(self, pos):
        # Check if the save button was clicked
        if self.save_button_rect.collidepoint(pos):
            save_board(self.config, self.board_state)
            print("\nBoard state saved.")
            return True # Signal to exit

        # Check if the next path button was clicked
        if self.next_path_button_rect.collidepoint(pos):
            self.next_path()
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