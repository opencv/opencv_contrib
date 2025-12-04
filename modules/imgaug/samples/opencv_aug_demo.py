import cv2
import copy


def random_crop(image):
    transform = cv2.imgaug.RandomCrop((300, 300))
    return transform.call(image)


def random_flip(image):
    transform = cv2.imgaug.RandomFlip(flipCode=1, p=0.8)
    return transform.call(image)


def center_crop(image):
    transform = cv2.imgaug.CenterCrop(size=(100, 100))
    return transform.call(image)


def pad(image):
    transform = cv2.imgaug.Pad(padding=(10, 10, 10, 10))
    return transform.call(image)


def random_resized_crop(image):
    transform = cv2.imgaug.RandomResizedCrop(size=(100, 100))
    return transform.call(image)


def compose(image):
    transform = cv2.imgaug.Compose([
        cv2.imgaug.Resize((1024, 1024)),
        cv2.imgaug.RandomCrop((800, 800)),
        cv2.imgaug.RandomFlip(),
        cv2.imgaug.CenterCrop((512, 512)),
    ])
    return transform.call(image)


def main():
    # read image
    input_path = "../../../samples/data/corridor.jpg"
    src = cv2.imread(input_path)

    while True:
        image = copy.copy(src)
        image = compose(image)
        cv2.imshow("dst", image)
        ch = cv2.waitKey(1000)
        if ch == 27:
            break


if __name__ == '__main__':
    main()
