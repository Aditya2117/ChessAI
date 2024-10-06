import pygame
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import json

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class ChessAI:
    def __init__(self):
        self.model = ChessNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.board = chess.Board()
        self.load_model()

    def board_to_input(self, board):
        x = torch.zeros(64 * 12)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                x[i * 12 + (piece.piece_type - 1) + (6 if piece.color else 0)] = 1
        return x

    def get_best_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = None
        best_value = float('-inf') if board.turn else float('inf')
        
        for move in legal_moves:
            board.push(move)
            value = self.model(self.board_to_input(board)).item()
            board.pop()
            
            if board.turn:  # White's turn (maximize)
                if value > best_value:
                    best_value = value
                    best_move = move
            else:  # Black's turn (minimize)
                if value < best_value:
                    best_value = value
                    best_move = move
        
        return best_move

    def train(self, board, result):
        self.optimizer.zero_grad()
        board_input = self.board_to_input(board)
        prediction = self.model(board_input)
        
        if result == '1-0':
            target = torch.tensor([1.0])
        elif result == '0-1':
            target = torch.tensor([-1.0])
        else:
            target = torch.tensor([0.0])
        
        loss = nn.MSELoss()(prediction, target)
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), 'chess_model.pth')

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load('chess_model.pth'))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No existing model found. Starting from scratch.")

class ChessGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((900, 700))
        self.clock = pygame.time.Clock()
        self.ai = ChessAI()
        self.board = chess.Board()
        self.square_size = 75
        self.board_offset = (50, 50)
        self.font = pygame.font.Font(None, 24)
        self.load_images()
        self.load_stats()

    def load_images(self):
        self.images = {}
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        for piece in pieces:
            self.images[piece] = pygame.transform.scale(pygame.image.load(f'images/{piece}B.png'), (self.square_size, self.square_size))
            self.images[piece.upper()] = pygame.transform.scale(pygame.image.load(f'images/{piece.upper()}W.png'), (self.square_size, self.square_size))

    def load_stats(self):
        try:
            with open('chess_stats.json', 'r') as f:
                self.stats = json.load(f)
        except FileNotFoundError:
            self.stats = {"games_played": 0, "wins": 0, "losses": 0, "draws": 0}

    def save_stats(self):
        with open('chess_stats.json', 'w') as f:
            json.dump(self.stats, f)

    def draw_board(self):
        colors = [(238, 238, 210), (118, 150, 86)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, (
                    self.board_offset[0] + col * self.square_size,
                    self.board_offset[1] + row * self.square_size,
                    self.square_size,
                    self.square_size
                ))

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x = self.board_offset[0] + (chess.square_file(square) * self.square_size)
                y = self.board_offset[1] + (7 - chess.square_rank(square)) * self.square_size
                self.screen.blit(self.images[piece.symbol()], (x, y))

    def main_menu(self):
        play_button = pygame.Rect(350, 250, 200, 50)
        train_button = pygame.Rect(350, 350, 200, 50)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_button.collidepoint(event.pos):
                        self.play_game()
                    elif train_button.collidepoint(event.pos):
                        self.train_ai()

            self.screen.fill((255, 255, 255))
            pygame.draw.rect(self.screen, (0, 255, 0), play_button)
            pygame.draw.rect(self.screen, (255, 0, 0), train_button)
            
            play_text = self.font.render('Play with AI', True, (0, 0, 0))
            train_text = self.font.render('Train AI', True, (0, 0, 0))
            
            self.screen.blit(play_text, (play_button.x + 50, play_button.y + 15))
            self.screen.blit(train_text, (train_button.x + 60, train_button.y + 15))
            
            pygame.display.flip()
            self.clock.tick(30)

    def play_game(self):
        selected_square = None
        game_over = False
        player_color = chess.WHITE

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and self.board.turn == player_color and not game_over:
                    x, y = event.pos
                    file = (x - self.board_offset[0]) // self.square_size
                    rank = 7 - (y - self.board_offset[1]) // self.square_size
                    if 0 <= file < 8 and 0 <= rank < 8:
                        square = chess.square(file, rank)
                        if selected_square is None:
                            selected_square = square
                        else:
                            move = chess.Move(selected_square, square)
                            if move in self.board.legal_moves:
                                self.board.push(move)
                                selected_square = None
                            else:
                                selected_square = square

            if not game_over and self.board.turn != player_color:
                ai_move = self.ai.get_best_move(self.board)
                if ai_move:
                    self.board.push(ai_move)

            self.screen.fill((255, 255, 255))
            self.draw_board()
            
            if selected_square is not None:
                pygame.draw.rect(self.screen, (255, 0, 0), (
                    self.board_offset[0] + (chess.square_file(selected_square) * self.square_size),
                    self.board_offset[1] + (7 - chess.square_rank(selected_square)) * self.square_size,
                    self.square_size,
                    self.square_size
                ), 3)

            pygame.display.flip()
            self.clock.tick(30)

            if self.board.is_game_over():
                if not game_over:
                    game_over = True
                    self.update_stats(self.board.result())
                    self.ai.train(self.board, self.board.result())
                    self.ai.save_model()
                choice = self.show_game_over_popup(self.board.result())
                if choice == "rematch":
                    self.board.reset()
                    game_over = False
                    player_color = not player_color  # Switch sides
                elif choice == "main_menu":
                    return

    def train_ai(self):
        stop_button = pygame.Rect(700, 50, 150, 50)
        speed_slider = pygame.Rect(700, 120, 150, 20)
        speed = 30  # Initial speed

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if stop_button.collidepoint(event.pos):
                        running = False
                    elif speed_slider.collidepoint(event.pos):
                        speed = (event.pos[0] - speed_slider.x) / speed_slider.width * 60 + 1

            self.board.reset()
            while not self.board.is_game_over():
                move = self.ai.get_best_move(self.board)
                if move:
                    self.board.push(move)
                
                self.screen.fill((255, 255, 255))
                self.draw_board()
                pygame.draw.rect(self.screen, (255, 0, 0), stop_button)
                stop_text = self.font.render('Stop Training', True, (0, 0, 0))
                self.screen.blit(stop_text, (stop_button.x + 10, stop_button.y + 15))

                pygame.draw.rect(self.screen, (0, 0, 0), speed_slider, 2)
                pygame.draw.rect(self.screen, (0, 255, 0), 
                                 (speed_slider.x, speed_slider.y, 
                                  (speed - 1) / 60 * speed_slider.width, speed_slider.height))
                speed_text = self.font.render(f'Speed: {speed:.1f}', True, (0, 0, 0))
                self.screen.blit(speed_text, (speed_slider.x, speed_slider.y + 25))

                stats_text = [
                    f"Games Played: {self.stats['games_played']}",
                    f"Wins: {self.stats['wins']}",
                    f"Losses: {self.stats['losses']}",
                    f"Draws: {self.stats['draws']}"
                ]
                for i, text in enumerate(stats_text):
                    surf = self.font.render(text, True, (0, 0, 0))
                    self.screen.blit(surf, (700, 200 + i * 30))

                pygame.display.flip()
                self.clock.tick(speed)

            self.update_stats(self.board.result())
            self.ai.train(self.board, self.board.result())
            self.ai.save_model()

        print(f"Training completed. Games played: {self.stats['games_played']}")

    def update_stats(self, result):
        self.stats['games_played'] += 1
        if result == "1-0":
            self.stats['wins'] += 1
        elif result == "0-1":
            self.stats['losses'] += 1
        elif result == "1/2-1/2":
            self.stats['draws'] += 1
        self.save_stats()

    def show_game_over_popup(self, result):
        popup = pygame.Surface((300, 250))
        popup.fill((200, 200, 200))
        pygame.draw.rect(popup, (0, 0, 0), popup.get_rect(), 2)

        font = pygame.font.Font(None, 36)
        if result == '1-0':
            text = font.render('White wins!', True, (0, 0, 0))
        elif result == '0-1':
            text = font.render('Black wins!', True, (0, 0, 0))
        else:
            text = font.render('Draw!', True, (0, 0, 0))

        popup.blit(text, (150 - text.get_width() // 2, 50))

        rematch_button = pygame.Rect(50, 120, 200, 40)
        main_menu_button = pygame.Rect(50, 180, 200, 40)

        pygame.draw.rect(popup, (0, 255, 0), rematch_button)
        pygame.draw.rect(popup, (255, 0, 0), main_menu_button)

        rematch_text = font.render('Rematch', True, (0, 0, 0))
        main_menu_text = font.render('Main Menu', True, (0, 0, 0))

        popup.blit(rematch_text, (rematch_button.x + 50, rematch_button.y + 10))
        popup.blit(main_menu_text, (main_menu_button.x + 40, main_menu_button.y + 10))

        self.screen.blit(popup, (300, 225))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return "quit"
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if rematch_button.collidepoint((event.pos[0] - 300, event.pos[1] - 225)):
                        return "rematch"
                    elif main_menu_button.collidepoint((event.pos[0] - 300, event.pos[1] - 225)):
                        return "main_menu"

    def run(self):
        self.main_menu()
        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()