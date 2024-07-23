from utils_pytorch import UniversalFactory
import models
import losses



class ModelFactory(UniversalFactory):
    classes = [
        models.UNet,
        models.WNet,
        models.RRUNet,
        models.RRWNetAll,
        models.RRWNet,
    ]


class LossesFactory(UniversalFactory):
    classes = [
        losses.BCE3Loss,
        losses.RRLoss,
    ]
