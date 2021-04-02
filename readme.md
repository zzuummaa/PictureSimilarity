# Picture similarity

Программа проверяет "похожесть" картинок.

Среда, в которой отлаживался код:

```cmd
OpenCV 3.4.14
MSVC 142
```

## Сборка

Открыть `Developer Command Prompt for VS 2019`, перейти в папку с проектом и выполнить следующие команды:

```cmd
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=%OpenCV_install_dir% -G "CodeBlocks - NMake Makefiles" ..
cmake --build .
```

## Запуск

Открыть `Developer Command Prompt for VS 2019`, перейти в папку `build` проекта и выполнить следующие команды:

```cmd
Path=%Path%;%OpenCV_install_dir%/%OpenCV_ARCH%/%OpenCV_RUNTIME%/bin;
./PictureSimilarity.exe < ../input.txt
```

## Список использованных источников

1. [OpenCV-Python Tutorials » Feature Detection and Description » Feature Matching](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)

2. [OpenCV-Python Tutorials » Feature Detection and Description » ORB (Oriented FAST and Rotated BRIEF)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)

3. [cv::BFMatcher Class Reference](https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html)

4. [StackOverflow: How to calculate % score from ORB algorithm?](https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm)
