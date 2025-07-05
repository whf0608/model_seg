import numpy as np
from albumentations import Compose
from albumentations import (AdvancedBlur,
                            Blur,
                            CLAHE,ChannelDropout,ChannelDropout,ChannelShuffle,#ColorJitter,
                            Downscale,
                            Emboss,Equalize,
                            FDA,FancyPCA,FromFloat,
                            GaussNoise,GaussianBlur,GlassBlur,GridDistortion,
                            HistogramMatching,HueSaturationValue,HorizontalFlip,
                            ISONoise,ImageCompression,InvertImg,IAAPerspective,#IAAAdditiveGaussianNoise,
                            IAASharpen, IAAEmboss,IAAPiecewiseAffine,
                            MedianBlur,MotionBlur,MultiplicativeNoise,Normalize,
                            PixelDistributionAdaptation,Posterize,RGBShift,RandomBrightnessContrast,
                            RandomFog,RandomGamma,RandomRain,RandomShadow,RandomSnow,RandomRotate90,
                            RandomSunFlare,RandomToneCurve,
                            Sharpen,Solarize,Superpixels,ShiftScaleRotate,
                            TemplateTransform,ToFloat,ToGray,ToSepia,Transpose,
                            UnsharpMask,
                            OpticalDistortion, OneOf,
                            Flip
                            )
def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            #IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

# image = np.ones((300, 300, 3), dtype=np.uint8)
# mask = np.ones((300, 300), dtype=np.uint8)
# mask1 = np.ones((300, 300), dtype=np.uint8)
# whatever_data = "my name"
# augmentation = strong_aug(p=0.9)
# data = {"image": image, "masks": [mask,mask1],"mask1": mask1, "whatever_data": whatever_data, "additional": "hello"}
# augmented = augmentation(**data)
# image, mask, mask1,whatever_data, additional = augmented["image"], augmented["masks"], augmented["mask1"],augmented["whatever_data"], augmented["additional"]