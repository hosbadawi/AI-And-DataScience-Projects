import librosa
import librosa.display
import numpy as np
import matplotlib   
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
plt.figure(figsize=(20, 20))
import pandas as pd
import io
import torch
from website import RegNetX32
from torchvision import transforms

class AIMDetection:
    def __init__(self,Model_path):
        """ Make prediction for given Audio File """
        self.model = RegNetX32().Build_Model()
        self.model.load_state_dict(torch.load(Model_path, map_location=torch.device('cpu')))
        self.model.eval() 
        self.langs= ["Arabic","Germany","English","Spanish","French","Italian","Portuguese"]
        


    def _Generate_MelSpec(self, audio_path):
        """Generate Mel_spectogram image from Audio file"""
        y, sr = librosa.load(audio_path ,duration = 8, sr=16000)
        y = librosa.util.fix_length(y, size=8*16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr, fmin=20)
        plt.figure(figsize=(12,6))
        plt.switch_backend('agg')
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                    x_axis='time', y_axis='mel', fmax=sr )
        plt.axis("off")

     
        buf = io.BytesIO()
        
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        image = Image.open(buf).resize((224, 224))

        # img_array = np.array(image)

        buf.close()
        plt.close()

        return image


    def Predict(self, Image_Path):
        PIL_img = self._Generate_MelSpec(Image_Path)
        pil_to_tensor = transforms.ToTensor()(PIL_img).unsqueeze_(0)
        prediction = self.model(pil_to_tensor)
        _,Pred =torch.max(prediction, dim=1)
        # prediction = self.model(pil_to_tensor)
        return self.langs[Pred.numpy()[0]]