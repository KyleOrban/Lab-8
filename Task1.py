import cv2

# Загрузка изображения
image_path = r'images/variant-2.png'
image = cv2.imread(image_path)

# Применение размытия по Гауссу
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Сохранение результата
cv2.imwrite('blurred_image.jpg', blurred_image)

# Показать оригинал и размытое изображение
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()