from image_processor import preprocess_image, draw_grey_image
from utils import create_tensor_from_file

tensors_list = create_tensor_from_file()
processed_image = preprocess_image(tensors_list[2])
draw_grey_image(processed_image)
