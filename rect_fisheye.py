from defisheye import Defisheye


def rectify_fisheye(img_path, img_out_path):
    """
    Rectifies a fisheye image using the Defisheye library.

    Args:
        img_path (str): The path to the input image.
        img_out_path (str): The path to save the output image.

    Returns:
        PIL.Image: The rectified image as a PIL Image object.
    """
    dtype = 'linear'
    format = 'fullframe'
    fov = 180
    pfov = 120

    obj = Defisheye(img_path, dtype=dtype, format=format, fov=fov, pfov=pfov)

    # To save image locally 
    obj.convert(outfile=img_out_path)

    # To use the converted image in memory
    new_image = obj.convert()

    return new_image


if __name__ == '__main__':
    img_path = '/workspaces/MeteorDection/test/merged.jpg'
    img_out_path = '/workspaces/MeteorDection/test/merged_rectified.jpg'
    rectify_fisheye(img_path, img_out_path)