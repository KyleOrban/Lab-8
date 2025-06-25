import cv2
import numpy as np
from collections import deque

# Пути к файлам
TEMPLATE_PATH = 'ref-point.jpg'
FLY_IMAGE_PATH = 'fly64.png'

# Загрузка шаблона и мухи
template = cv2.imread(TEMPLATE_PATH, 0)
fly_img = cv2.imread(FLY_IMAGE_PATH, cv2.IMREAD_UNCHANGED)  # сохраняем альфа-канал

# ORB + Matcher
orb = cv2.ORB_create()
kp_template, des_template = orb.detectAndCompute(template, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Видеопоток
cap = cv2.VideoCapture(0)
coord_file = open("coords.txt", "w")

# Очередь для сглаживания
smooth_window = deque(maxlen=5)

def overlay_image_alpha(img, img_overlay, x, y):
    overlay_h, overlay_w = img_overlay.shape[:2]
    if x < 0 or y < 0 or x + overlay_w > img.shape[1] or y + overlay_h > img.shape[0]:
        return  # выходит за пределы кадра

    # Разделяем каналы
    b, g, r, a = cv2.split(img_overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0

    roi = img[y:y+overlay_h, x:x+overlay_w]
    img[y:y+overlay_h, x:x+overlay_w] = (roi * (1 - mask) + overlay_rgb * mask).astype(np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is not None:
        matches = bf.match(des_template, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 5:
            best_matches = matches[:10]
            points = [kp_frame[m.trainIdx].pt for m in best_matches]
            avg_x = int(np.mean([p[0] for p in points]))
            avg_y = int(np.mean([p[1] for p in points]))
            smooth_window.append((avg_x, avg_y))

            smooth_x = int(np.mean([p[0] for p in smooth_window]))
            smooth_y = int(np.mean([p[1] for p in smooth_window]))

            coord_file.write(f"{smooth_x}, {smooth_y}\n")

            # Положение для наложения мухи (по центру)
            fly_h, fly_w = fly_img.shape[:2]
            top_left_x = smooth_x - fly_w // 2
            top_left_y = smooth_y - fly_h // 2

            overlay_image_alpha(frame, fly_img, top_left_x, top_left_y)

    cv2.imshow('Tracking with Fly', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
coord_file.close()
cv2.destroyAllWindows()
