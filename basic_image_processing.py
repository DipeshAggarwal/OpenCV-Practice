import numpy as np
import argparse
import cv2

def brightness_control(og, by, brightness):
    """
    Returns an cv2 numpy array.

    og: Original array to be modified.
    by: Array to modify the original by.
    brightness: Integer. Amount by which to change the brightness.
    """

    if brightness > 0:
        return cv2.add(og, by)
    else:
        return cv2.subtract(og, by)

def reorder_channels(R, G, B, order):
    """
    Returns an cv2 numpy array.

    R: NumPy Array. Red Channel.
    G: NumPy Array. Green Channel.
    B: NumPy Array. Blue Channel.
    order: String. The new order of the final array/image.
    """

    if order == "bgr":
        return cv2.merge([B, G, R])
    elif order == "rgb":
        return cv2.merge([R, G, B])
    elif order == "rbg":
        return cv2.merge([R, B, G])
    elif order == "brg":
        return cv2.merge([B, R, G])
    elif order == "grb":
        return cv2.merge([G, R, B])
    elif order == "gbr":
        return cv2.merge([G, B, R])

# Set a few arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-o", "--output", help="Name of the saved image. Only required if an output file needs to be generated.")
ap.add_argument("-c", "--crop", nargs=4, type=int, help="Crop the image. Should be four numbers in the form start_x start_y end_x end_y")
ap.add_argument("-s", "--scale", type=float, default=1.0, help="Resize image.")
ap.add_argument("-r", "--rotate", type=int, help="Angle to rotate the image. To rotate image clockwise, provide a negative angle.")
ap.add_argument("-ra", "--rotate-around", nargs=2, type=int, help="Point to rotate the image around.")
ap.add_argument("-b", "--brightness", type=int, help="Change brightness level. Positive value will increase brightness and negative value will decrease it")
ap.add_argument("-ch", "--channels", default="bgr", help="Reorder channel of an image. Provide the three channels without separating them, ie, rgb")
ap.add_argument("-in", "--inverse", action="store_true", help="Inverses the image colours.")
ap.add_argument("-br", "--brighten-red", action="store_true", help="Only brightens the red channel.")
ap.add_argument("-bg", "--brighten-green", action="store_true", help="Only brightens the green channel.")
ap.add_argument("-bb", "--brighten-blue", action="store_true", help="Only brightens the blue channel.")
ap.add_argument("-ir", "--inverse-red", action="store_true", help="Only inverses the red channel.")
ap.add_argument("-ig", "--inverse-green", action="store_true", help="Only inverses the green channel.")
ap.add_argument("-ib", "--inverse-blue", action="store_true", help="Only inverses the blue channel.")
ap.add_argument("-cf", "--crop-first", action="store_true", help="Provide if crop needs to happen before all operations otherwise it happens in the last")

# Read the provided arguments and store them
args = vars(ap.parse_args())

if not args["brighten_red"] and not args["brighten_blue"] and not args["brighten_green"]:
    brighten_full = True
elif args["brighten_red"] and args["brighten_blue"] and args["brighten_green"]:
    brighten_full = True
else:
    brighten_full = False

if args["inverse"]:
    inverse_full = True
elif not args["inverse_red"] and not args["inverse_blue"] and not args["inverse_green"]:
    inverse_full = True
elif args["inverse_red"] and args["inverse_blue"] and args["inverse_green"]:
    inverse_full = True
else:
    inverse_full = False

image = cv2.imread(args["image"])
h, w = image.shape[:2]

if args["crop_first"] and args["crop"]:
    sx, sy, ex, ey = args["crop"]
    image = image[sy:ey, sx:ex]

if args["brightness"]:
    if brighten_full:
        by = np.ones(image.shape, dtype="uint8") * abs(args["brightness"])
        image = brightness_control(image, by, args["brightness"])
    else:
        B, G, R = cv2.split(image)
        by = np.ones([h, w], dtype="uint8") * abs(args["brightness"])

        if args["brighten_red"]:
            R = brightness_control(R, by, args["brightness"])
        
        if args["brighten_blue"]:
            B = brightness_control(B, by, args["brightness"])
        
        if args["brighten_green"]:
            G = brightness_control(G, by, args["brightness"])

        image = reorder_channels(R, G, B, args["channels"])

if args["scale"] and not args["rotate"]:
    w = int(w * args["scale"])
    h = int(h * args["scale"])
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

elif args["rotate"]:
    if args["rotate_around"]:
        rx, ry = args["rotate_around"]
    else:
        rx, ry = w // 2, h // 2
    M = cv2.getRotationMatrix2D((rx, ry), args["rotate"], args["scale"])
    image = cv2.warpAffine(image, M, (w, h))

if not args["crop_first"] and args["crop"]:
    sx, sy, ex, ey = args["crop"]
    image = image[sy:ey, sx:ex]

if inverse_full:
    image = cv2.bitwise_not(image)
else:
    B, G, R = cv2.split(image)

    if args["inverse_red"]:
        R = cv2.bitwise_not(R)
    if args["inverse_green"]:
        G = cv2.bitwise_not(G)
    if args["inverse_blue"]:
        B = cv2.bitwise_not(B)

    image = reorder_channels(R, G, B, args["channels"])

if brighten_full:
    B, G, R = cv2.split(image)
    image = reorder_channels(R, G, B, args["channels"])

cv2.imshow("Processed Image", image)
if args["output"]:
    cv2.imwrite(args["output"] + ".jpg", image)
cv2.waitKey(0)
