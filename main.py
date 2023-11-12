from my_funcs import *

model = VGG16(weights='imagenet', include_top=False)

image_folder = 'Tic-tac-toe game images/'
image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

output_folder = 'results'
# Crate new if not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in image_files:
    # Зчитування зображення
    image = cv2.imread(image_file)
    bounding_boxes = get_image_bounding_boxes(image)

    dict_with_images_boxes = label_bounding_boxes(image, bounding_boxes, model)
    lines = get_winner_and_draw_line(dict_with_images_boxes)
    x1, y1, x2, y2 = lines

    #Draw lines
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 5)
    plt.imshow(image)
    plt.show()

    #Save
    output_file = os.path.join(output_folder, os.path.basename(image_file))
    cv2.imwrite(output_file, image)