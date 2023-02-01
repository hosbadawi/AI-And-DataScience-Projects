import torchvision.models as models
import torch.nn as nn

class VGGModel:
    def Build_Model(self):

        model = models.vgg16(pretrained=False)

        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 7),
            nn.LogSoftmax(dim=1),
        )

        return model