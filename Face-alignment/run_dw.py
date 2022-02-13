import sys
import face_alignment
from skimage import io

if __name__ == "__main__":

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    input = io.imread(sys.argv[1])
    preds = fa.get_landmarks(input)

    import numpy as np
    print(type(preds))
    print(sys.argv[2])
    np.array(preds).tofile(sys.argv[2])

    #print(preds)
