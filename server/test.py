import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import test5.a as a

print(a.calc(1,2))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "Face-alignment"))
print(os.path.dirname(os.path.join(os.path.abspath(os.path.dirname(__file__))), "Face-alignment"))
import face_alignment