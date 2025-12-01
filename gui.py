import pygame
from PIL import Image
import cv2

def draw_button(screen, rect, text, font, hover=False, BUTTON_COLOR=(70,70,70),
                BUTTON_HOVER=(100,100,100), TEXT_COLOR=(255,255,255)):
    color = BUTTON_HOVER if hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect)
    txt_surf = font.render(text, True, TEXT_COLOR)
    txt_rect = txt_surf.get_rect(center=rect.center)
    screen.blit(txt_surf, txt_rect)

def create_surface_from_cv2(img):
    return pygame.surfarray.make_surface(cv2.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

def capture_gif_frame(src_img, evo_img):
    target_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    evo_rgb = cv2.cvtColor(evo_img, cv2.COLOR_BGR2RGB)
    combined = cv2.hconcat([target_rgb, evo_rgb])
    return Image.fromarray(combined)
