# Picture similarity

Программа проверяет "похожесть" картинок.

Среда, в которой отлаживался код:

```cmd
OpenCV 3.4.14
MSVC 142
```

## Сборка

```cmd
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=%OpenCV_install_dir% -G "CodeBlocks - NMake Makefiles" ..
cmake --build .
```

## Запуск

```cmd
Path=%Path%;%OpenCV_install_dir%/%OpenCV_ARCH%/%OpenCV_RUNTIME%/bin;
./PictureSimilarity.exe
```

## Список использованных источников

1. [OpenCV-Python Tutorials » Feature Detection and Description » Feature Matching](https://docs.opencv.org/master/db/deb/tutorial_display_image.html)

2. [OpenCV-Python Tutorials » Feature Detection and Description » ORB (Oriented FAST and Rotated BRIEF)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)

3. [cv::BFMatcher Class Reference](https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html)

4. [StackOverflow: How to calculate % score from ORB algorithm?](https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm)
