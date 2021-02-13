# from extractor import extractor
import cv2
from Extractor import ObjectExtractor


def main():
    image_path = '/home/ziz/school/LOG795/object_extraction/pomme.jpg'
    image = cv2.imread(image_path)
    objects = ObjectExtractor().extract(image)
    # extractor().test(image_path)
    # print(objects)

if __name__ == '__main__':
    main()
    