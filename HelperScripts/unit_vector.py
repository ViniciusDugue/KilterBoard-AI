import pygame
import numpy as np
import pyperclip
# Initialize pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rotate Line")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Line parameters
line_length = 150
line_thickness = 2
line_color = BLACK

# Origin point
origin = np.array([width // 2, height // 2])

# Initial angle
angle = 0

# Font for displaying angle
font = pygame.font.Font(None, 36)

# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    # Draw the origin point
    pygame.draw.circle(screen, BLACK, origin.astype(int), 5)
    
    # Calculate end point of the line
    end_point = origin + np.array([line_length * np.cos(np.radians(angle)),
                                   line_length * np.sin(np.radians(angle))])
    
    # Draw the rotating line
    pygame.draw.line(screen, line_color, origin.astype(int), end_point.astype(int), line_thickness)
    
    # Calculate end points of the perpendicular line
    perpendicular_length = 200
    perpendicular_start = end_point - perpendicular_length/2 * np.array([-np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    perpendicular_end = perpendicular_start + perpendicular_length * np.array([-np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    
    # Draw the perpendicular line
    pygame.draw.line(screen, RED, perpendicular_start.astype(int), perpendicular_end.astype(int), line_thickness)

    # Display angle
    angle_text = font.render(f"Angle: {(360- angle) % 360} degrees", True, BLACK)
    screen.blit(angle_text, (10, 10))
    
    # Calculate unit vector
    line_vector = np.array([np.cos(np.radians(angle)), -np.sin(np.radians(angle))])
    unit_vector = line_vector / np.linalg.norm(line_vector)

    # Round the components of the unit vector to the nearest tenth decimal place
    rounded_unit_vector = np.round(unit_vector, decimals=2)

    # Display unit vector
    unit_vector_text = font.render(f"Unit Vector: {rounded_unit_vector}", True, BLACK)
    screen.blit(unit_vector_text, (10, 50))
    
    # Update the display
    pygame.display.flip()
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Calculate angle between origin and mouse position
            mouse_pos = np.array(pygame.mouse.get_pos())
            diff = mouse_pos - origin
            angle = np.degrees(np.arctan2(diff[1], diff[0]))
            if angle < 0:
                angle += 360
    keys = pygame.key.get_pressed()
    if keys[pygame.K_c]:
        rounded_unit_vector_str = f"{rounded_unit_vector[0]},{rounded_unit_vector[1]}"
        pyperclip.copy(rounded_unit_vector_str)
        print("Rounded unit vector copied to clipboard:")

# Quit pygame
pygame.quit()
