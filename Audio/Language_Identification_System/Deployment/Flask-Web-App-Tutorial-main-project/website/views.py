
from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
plt.figure(figsize=(20, 20))
import pandas as pd
import io
import torch
from website import VGGModel
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn 
from .models import Note
from . import db
import json
import os 
from flask import Flask, render_template,flash, request, redirect
import winsound
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
class AIMDetection:
    def __init__(self,Model_path):
        """ Make prediction for given Audio File """
        self.model = VGGModel().Build_Model()
        self.model.load_state_dict(torch.load(Model_path, map_location=torch.device('cpu')))
        self.model.eval() 
        self.langs= ["Arabic","Germany","English","Spanish","French","Italian","Portuguese"]
        


    def _Generate_MelSpec(self, audio_path):
        """Generate Mel_spectogram image from Audio file"""
        y, sr = librosa.load(audio_path ,duration = 8, sr=16000)
        y = librosa.util.fix_length(y, size=8*16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr, fmin=20)
        
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

views = Blueprint('views', __name__)


infer = AIMDetection("CNN_1D.pt")
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        

        if "file" not in request.files:
            flash('Choose File First', category='success') 
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file.save(os.path.join('upload_sounds', file.filename))
            res=infer.Predict("./upload_sounds/"+ file.filename)
            new_note = Note(data=res, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Language Detected!', category='success')    
            
            # winsound.PlaySound("./upload_sounds/"+ file.filename, winsound.SND_ASYNC | winsound.SND_ALIAS )

        else:
             print("not a wav file")
             flash('File Not Accepted check the file type', category='success') 
           
            

    return render_template("home.html", user=current_user)


@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})


