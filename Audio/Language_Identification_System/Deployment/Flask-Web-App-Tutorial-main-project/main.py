from website import create_app
from  website import AIMDetection
from website import VGGModel 
import sys
sys.path.append('/website/')

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
