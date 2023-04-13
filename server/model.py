from fastai.vision import *
from fastai.vision.all import *
import pathlib

class Model:
    def __init__(self, model_path):
        '''
        EXPORT_PATH = pathlib.Path(model_path)

        @contextmanager
        def set_posix_windows():
            posix_backup = pathlib.PosixPath
            try:
                pathlib.PosixPath = pathlib.WindowsPath
                yield
            finally:
                pathlib.PosixPath = posix_backup
        
        with set_posix_windows():
            self.learn = load_learner(EXPORT_PATH, cpu=True)
        '''
        self.learn = load_learner(model_path, cpu=True)

    def predict(self, img):
        img = img.resize((224,224))
        prediction,_,_ = self.learn.predict(img)
        return prediction