import cv2
import numpy as np
import math
import random

# Fitness
def mse_downsampled(candidate_img, src_down, ds):
    cand_small = cv2.resize(candidate_img, (ds, ds),
                            interpolation=cv2.INTER_AREA).astype(np.float32)
    diff = cand_small - src_down
    return float(np.mean(diff * diff))

# Mutation
def mutate_image(parent_img_uint8, generation_idx, colors, h, w):
    img = np.copy(parent_img_uint8)
    clr = random.choice(colors)
    
    # Adaptive size scaling
    size_factor_h = max(1, int(h // (5 + generation_idx**0.4)))
    size_factor_w = max(1, int(w // (5 + generation_idx**0.4)))
    s1 = random.randint(0, size_factor_h)
    s2 = random.randint(0, size_factor_w)
    
    lt = random.randint(0, h-1)
    lg = random.randint(0, w-1)
    rot = random.randint(0, 360)
    
    # Mutation types
    mutation_type = random.choice(["ellipse", "rectangle", "line"])
    if mutation_type == "ellipse":
        cv2.ellipse(img, (lg, lt), (s1, s2), rot, 0, 360, clr, -1)
    elif mutation_type == "rectangle":
        cv2.rectangle(img, (lg, lt), (lg+s2, lt+s1), clr, -1)
    elif mutation_type == "line":
        cv2.line(img, (lg, lt), (lg+s2, lt+s1), clr, max(1, s1//2))
    
    return img, (clr, rot, lt, lg, s1, s2, mutation_type)

# Crossover
def crossover(img1, img2, num_patches=20, patch_size_range=(10, 30)):
    h, w = img1.shape[:2]
    blended = img1.copy()

    for _ in range(num_patches):
        ph = random.randint(*patch_size_range)
        pw = random.randint(*patch_size_range)
        y = random.randint(0, h - ph)
        x = random.randint(0, w - pw)

        blended[y:y+ph, x:x+pw] = img2[y:y+ph, x:x+pw]

    return blended


