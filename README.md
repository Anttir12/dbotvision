
To get opencv in venv. Install it system-wide first.

`sudo apt install libopencv-dev python3-opencv`

test it works (not in venv):
`python3 -c "import cv2; print(cv2.__version__)"`

(note that version and path might vary depending on your system etc.)
Then copy /usr/lib/python3/dist-packages/cv2.cpython-38-x86_64-linux-gnu.so
to your venv
<project-root>/venv/lib/python3.8/site-packages/.