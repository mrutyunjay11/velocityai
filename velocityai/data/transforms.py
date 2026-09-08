class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor:
    """Convert a numpy.ndarray to velocityai tensor."""
    def __call__(self, pic):
        import velocityai as vai
        return vai.tensor(pic)
