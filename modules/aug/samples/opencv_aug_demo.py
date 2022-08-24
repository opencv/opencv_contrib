import cv2
import matplotlib.pyplot as plt


def random_crop(image):
    transform = cv2.RandomCrop((300, 300))
    return transform.call(image)


def random_flip(image):
    transform = cv2.RandomFlip(flipCode=1, p=0.8)
    return transform.call(image)


def center_crop(image):
    transform = cv2.CenterCrop(size=(100, 100))
    return transform.call(image)


def pad(image):
    transform = cv2.Pad(padding=(10, 10, 10, 10))
    return transform.call(image)


def random_resized_crop(image):
    transform = cv2.RandomResizedCrop(size=(100, 100))
    return transform.call(image)


def compose(image):
    transform = cv2.Compose([
        cv2.RandomCrop((300, 300)),
        cv2.RandomFlip(),
        cv2.CenterCrop((100, 200)),
    ])
    return transform.call(image)


def main():
    # read image
    input_path = "../../../samples/data/corridor.jpg"
    image = cv2.imread(input_path)

    image = compose(image)

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
