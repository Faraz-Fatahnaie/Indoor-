from torchvision.models import efficientnet_v2_s


def efficientnet_b0(**kwargs):
    """
    Simple EfficientNetB0
    :param kwargs:
    :return:
    """

    return efficientnet_v2_s(spatial_dims=2, in_channels=3, num_classes=67, dropout=0.3)
