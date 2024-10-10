import pygame
import numpy as np
from scipy.io import wavfile
import pygame.sndarray
import os
import random

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Anonymous Hacker Voice Sync")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
BACKGROUND_COLOR = (30, 30, 30)  # Dark background
MATRIX_GREEN = (0, 255, 0)

# Font for matrix effect
matrix_font = pygame.font.Font(None, 14)

# Matrix code characters
matrix_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%^&*()_+-=[]{}|;:,.<>?"

# Matrix code streams
class MatrixStream:
    def __init__(self, x, y, rainbow=False):
        self.x = x
        self.y = y
        self.speed = random.randint(1, 5)
        self.chars = [random.choice(matrix_characters) for _ in range(30)]
        self.rainbow = rainbow
        self.colors = self.generate_colors()

    def generate_colors(self):
        if self.rainbow:
            return [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(30)]
        else:
            return [(0, random.randint(50, 255), 0) for _ in range(30)]

    def update(self):
        self.y += self.speed
        if self.y > HEIGHT:
            self.y = random.randint(-100, -20)
        self.chars.pop()
        self.chars.insert(0, random.choice(matrix_characters))
        self.colors.pop()
        if self.rainbow:
            self.colors.insert(0, (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
        else:
            self.colors.insert(0, (0, random.randint(50, 255), 0))

    def draw(self, surface):
        for i, (char, color) in enumerate(zip(self.chars, self.colors)):
            text = matrix_font.render(char, True, color)
            surface.blit(text, (self.x, self.y + i * 14))

# Create matrix streams
matrix_streams = [MatrixStream(x * 20, random.randint(-500, 0)) for x in range(WIDTH // 20)]
rainbow_matrix_streams = [MatrixStream(x * 20, random.randint(-500, 0), rainbow=True) for x in range(WIDTH // 20)]

# Load the audio file
audio_file = "A.wav"
sample_rate, audio_data = wavfile.read(audio_file)

# Convert to stereo if mono
if len(audio_data.shape) == 1:
    audio_data = np.column_stack((audio_data, audio_data))
else:
    audio_data = audio_data[:, :2]

# Normalize audio data to 16-bit range
audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)

# Create a Pygame sound object
sound = pygame.sndarray.make_sound(audio_data)

# Adjustable parameters
sensitivity = 5000
max_mouth_height = 20
min_mouth_height = 5

def get_current_amplitude(start, chunk_size):
    end = start + chunk_size
    if end > len(audio_data):
        return 0
    chunk = audio_data[start:end, 0]
    return np.max(np.abs(chunk))

def map_amplitude_to_mouth(amplitude):
    normalized_amp = min(amplitude / sensitivity, 1.0)
    return int(normalized_amp ** 2 * (max_mouth_height - min_mouth_height) + min_mouth_height)

def draw_hacker(screen, mouth_height):
    # Draw body
    pygame.draw.rect(screen, DARK_GRAY, (WIDTH//2 - 75, HEIGHT//2 + 100, 150, 200))  # Torso
    pygame.draw.rect(screen, DARK_GRAY, (WIDTH//2 - 100, HEIGHT//2 + 100, 25, 150))  # Left arm
    pygame.draw.rect(screen, DARK_GRAY, (WIDTH//2 + 75, HEIGHT//2 + 100, 25, 150))   # Right arm

    # Draw hoodie
    pygame.draw.ellipse(screen, DARK_GRAY, (WIDTH//2 - 120, HEIGHT//2 - 150, 240, 300))
    
    # Draw face shadow
    pygame.draw.ellipse(screen, GRAY, (WIDTH//2 - 100, HEIGHT//2 - 100, 200, 220))
    
    # Draw eyes (as shadow)
    pygame.draw.ellipse(screen, BLACK, (WIDTH//2 - 60, HEIGHT//2 - 40, 40, 20))
    pygame.draw.ellipse(screen, BLACK, (WIDTH//2 + 20, HEIGHT//2 - 40, 40, 20))
    
    # Draw nose shadow
    pygame.draw.polygon(screen, DARK_GRAY, [(WIDTH//2, HEIGHT//2 + 10), (WIDTH//2 - 10, HEIGHT//2 + 30), (WIDTH//2 + 10, HEIGHT//2 + 30)])
    
    # Draw mouth
    pygame.draw.rect(screen, BLACK, (WIDTH//2 - 40, HEIGHT//2 + 50, 80, mouth_height))

# Main game loop
running = True
clock = pygame.time.Clock()
sound.play()
start_time = pygame.time.get_ticks()
chunk_size = 1024
audio_length = len(audio_data) / sample_rate * 1000

font = pygame.font.Font(None, 24)

# Background mode
background_mode = 0  # 0: Classic, 1: Matrix, 2: Rainbow Matrix

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                sensitivity += 500
            elif event.key == pygame.K_DOWN:
                sensitivity = max(500, sensitivity - 500)
            elif event.key == pygame.K_RIGHT:
                max_mouth_height = min(40, max_mouth_height + 5)
            elif event.key == pygame.K_LEFT:
                max_mouth_height = max(10, max_mouth_height - 5)
            elif event.key == pygame.K_m:
                background_mode = (background_mode + 1) % 3
            elif event.key == pygame.K_r and background_mode == 0:
                BACKGROUND_COLOR = (BACKGROUND_COLOR[0] + 20 % 256, BACKGROUND_COLOR[1], BACKGROUND_COLOR[2])
            elif event.key == pygame.K_g and background_mode == 0:
                BACKGROUND_COLOR = (BACKGROUND_COLOR[0], BACKGROUND_COLOR[1] + 20 % 256, BACKGROUND_COLOR[2])
            elif event.key == pygame.K_b and background_mode == 0:
                BACKGROUND_COLOR = (BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2] + 20 % 256)

    # Draw background
    if background_mode == 0:  # Classic
        screen.fill(BACKGROUND_COLOR)
    elif background_mode == 1:  # Matrix
        screen.fill(BLACK)
        for stream in matrix_streams:
            stream.update()
            stream.draw(screen)
    else:  # Rainbow Matrix
        screen.fill(BLACK)
        for stream in rainbow_matrix_streams:
            stream.update()
            stream.draw(screen)

    current_time = pygame.time.get_ticks() - start_time

    if current_time > audio_length:
        mouth_height = min_mouth_height
    else:
        current_sample = int(current_time * sample_rate / 1000)
        amplitude = get_current_amplitude(current_sample, chunk_size)
        mouth_height = map_amplitude_to_mouth(amplitude)

    # Draw the hacker face and body
    draw_hacker(screen, mouth_height)

    # Draw volume meter
    meter_height = int((amplitude / sensitivity) * 100)
    pygame.draw.rect(screen, (255, 0, 0), (10, HEIGHT - 10 - meter_height, 20, meter_height))

    # Display current settings
    text = font.render(f"Sensitivity: {sensitivity}, Max Height: {max_mouth_height}", True, WHITE)
    screen.blit(text, (10, 10))
    if background_mode == 0:
        text = font.render(f"BG Color: R:{BACKGROUND_COLOR[0]} G:{BACKGROUND_COLOR[1]} B:{BACKGROUND_COLOR[2]}", True, WHITE)
        screen.blit(text, (10, 40))
    mode_names = ["Classic", "Matrix", "Rainbow Matrix"]
    mode_text = font.render(f"Mode: {mode_names[background_mode]} (Press M to cycle)", True, WHITE)
    screen.blit(mode_text, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()