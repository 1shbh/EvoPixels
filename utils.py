import pickle
from os import mkdir
from os.path import isfile, isdir, join, split, splitext
from datetime import datetime
import cv2

def save_to_disk(logs, img_path, final_img):
    base_path, file = split(img_path)
    fname, _ = splitext(file)
    folders = {"logs": "txt", "results": "png"}

    for folder, ext in folders.items():
        folder_dir = join(base_path, folder)
        output_dir = join(folder_dir, f"{fname}.{ext}")
        if not isdir(folder_dir):
            mkdir(folder_dir)
        if isfile(output_dir):
            now = datetime.now()
            dt_string = now.strftime("%H%M%S")
            new_fname = fname + dt_string + "." + ext
            output_dir = join(base_path, folder, new_fname)

        if folder == "logs":
            with open(output_dir, "wb+") as f:
                pickle.dump(logs, f)
            print(f"[+] Saved logs at {output_dir}")
        else:
            cv2.imwrite(output_dir, final_img)
            print(f"[+] Saved final image at {output_dir}")
