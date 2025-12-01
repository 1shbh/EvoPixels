import pygame
import cv2
import random
import os

from config import *
from canvas import Canvas
from config import MAX_IMG_WIDTH, MAX_IMG_HEIGHT
from evolution import mutate_image, crossover, mse_downsampled
from gui import draw_button, create_surface_from_cv2, capture_gif_frame
from utils import save_to_disk


def get_next_gif_name(base="evolution"):
    folder = "output"
    if not os.path.exists(folder):
        os.makedirs(folder)

    i = 1
    while True:
        fname = os.path.join(folder, f"{base}_{i}.gif")
        if not os.path.exists(fname):
            return fname
        i += 1

# Pygame init
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Evolutionary Image Reconstruction")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# Load image
src_img = cv2.imread(IMAGE_PATH)
if src_img is None:
    raise FileNotFoundError(f"Cannot open {IMAGE_PATH}")

h, w = src_img.shape[:2]
scale = min(MAX_IMG_WIDTH / w, MAX_IMG_HEIGHT / h, 1.0)
new_w = int(w * scale)
new_h = int(h * scale)
src_img = cv2.resize(src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
cvs = Canvas(src_img, K, DS_FIT)

# Initialize population
population = [(cvs.blank_arr.copy(), cvs.blank_MSE)]
for _ in range(POPULATION_SIZE - 1):
    mutated = cvs.blank_arr.copy()
    clr = random.choice(cvs.colors)
    cv2.circle(
        mutated,
        (random.randint(0, cvs.w - 1), random.randint(0, cvs.h - 1)),
        random.randint(5, max(5, min(cvs.h, cvs.w) // 10)),
        clr,
        -1
    )
    mse = mse_downsampled(mutated, cvs.src_down, cvs.ds)
    population.append((mutated, mse))

darwin_logs = []
paused = True
step_mode = False
generation = 0
gif_frames = []

# Buttons
play_btn = pygame.Rect(20, WINDOW_HEIGHT - BUTTON_HEIGHT - 15, 100, BUTTON_HEIGHT)
pause_btn = pygame.Rect(130, WINDOW_HEIGHT - BUTTON_HEIGHT - 15, 100, BUTTON_HEIGHT)
step_btn = pygame.Rect(240, WINDOW_HEIGHT - BUTTON_HEIGHT - 15, 100, BUTTON_HEIGHT)
gif_btn = pygame.Rect(350, WINDOW_HEIGHT - BUTTON_HEIGHT - 15, 150, BUTTON_HEIGHT)

running = True
while running:

    screen.fill(BG_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if play_btn.collidepoint(mx, my): paused = False
            if pause_btn.collidepoint(mx, my): paused = True
            if step_btn.collidepoint(mx, my): step_mode = True
            if gif_btn.collidepoint(mx,my) and gif_frames:
                save_name = get_next_gif_name("evolution")
                gif_frames[0].save(save_name, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
                print(f"[+] GIF saved as {save_name}")

    # Evolution Step
    if not paused or step_mode:

        generation += 1

        if generation >= N_GENERATIONS:
            paused = True
            print("[+] Reached max generations")
            

        new_population = []

        # Sort and define best
        population.sort(key=lambda x: x[1])
        best = population[0]  


        # CROSSOVER
        best_parents = population[:5]
        

        for _ in range(NUM_CROSSOVERS):
            p1, _ = random.choice(best_parents)
            p2, _ = random.choice(best_parents)

            if p1 is p2:
                continue

            child_img = crossover(p1, p2)
            child_mse = mse_downsampled(child_img, cvs.src_down, cvs.ds)
            new_population.append((child_img, child_mse))

        # MUTATION
        colors = cvs.colors
        h, w = cvs.h, cvs.w
        pop_pref = population[:3] if len(population) >= 3 else population

        for i in range(M_CANDIDATES - NUM_CROSSOVERS):
            parent_img, _ = random.choice(pop_pref)
            mutated_img, m_vars = mutate_image(parent_img, generation, colors, h, w)
            m_mse = mse_downsampled(mutated_img, cvs.src_down, cvs.ds)
            new_population.append((mutated_img, m_mse))
            darwin_logs.append((m_vars, m_mse))

        # Elitism
        new_population.append((best[0].copy(), best[1]))

        new_population.sort(key=lambda x: x[1])
        population = new_population[:POPULATION_SIZE]

        step_mode = False

        # GIF frame capture
        if generation % GIF_FRAME_SKIP == 0:
            gif_frames.append(capture_gif_frame(src_img, population[0][0]))

    # images
    target_surf = create_surface_from_cv2(src_img)
    evo_surf = create_surface_from_cv2(population[0][0])

    screen.blit(pygame.transform.scale(target_surf, (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)), (0, 0))
    screen.blit(pygame.transform.scale(evo_surf, (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)), (WINDOW_WIDTH // 2, 0))

    # buttons
    draw_button(screen, play_btn, "Play", font, play_btn.collidepoint(mouse_pos))
    draw_button(screen, pause_btn, "Pause", font, pause_btn.collidepoint(mouse_pos))
    draw_button(screen, step_btn, "Step", font, step_btn.collidepoint(mouse_pos))
    draw_button(screen, gif_btn, "Save GIF", font, gif_btn.collidepoint(mouse_pos))

    # Generation info
    gen_text = font.render(f"Generation: {generation} | Best MSE: {int(population[0][1])}", True, (255, 255, 255))
    screen.blit(gen_text, (520, WINDOW_HEIGHT - BUTTON_HEIGHT - 5))

    pygame.display.flip()
    clock.tick(60)

# Save final result
save_to_disk(darwin_logs, IMAGE_PATH, population[0][0])



pygame.quit()




