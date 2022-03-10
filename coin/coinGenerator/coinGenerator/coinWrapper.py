import ctypes
import enum
import json


lib = ctypes.cdll.LoadLibrary('./lib.so')

# 512 고정 생각하삼

arg = [
    "",  # 첫번째 일단 빈 string으로 두삼 arg가 한 칸씩 밀림
    "007F.png",  # 원본
    "NONE",  # 마스크
    "NONE",  # parsing
    "007F_TEXT.png",  # 텍스트 이미지
    "007_2.stl",  # 출력파일명
    "003R.png",  # 뒷면 이미지
]

length = len(arg)

argv_type = (ctypes.c_char_p * length)
argv_select = argv_type()

for i, val in enumerate(arg):
    argv_select[i] = val.encode('utf-8')

lib.main.argtypes = [ctypes.c_int, argv_type]
print(lib.main(length, argv_select))
