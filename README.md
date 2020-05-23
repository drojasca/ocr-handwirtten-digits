# OCR Hand-Written Digits 
## Daniel Rojas
---------------------------------------------------------
# DESCRIPTION
This project was done to familiarize myself with ROS and OpenCV.

Using an SVM run on a ROS node, a model was trained to predict the value of a hand written digit based
on a Histogram of Oriented Gradients (HOG).
By passing an image with handwritten digits on a plain, light background, the node is
able to isolate the dark areas and pass them to the SVM for prediction.

The annotated image is then sent and displayed in RQT using 
[rqt image view](http://wiki.ros.org/rqt_image_view)

---------------------------------------------------------
# DEPENDENCIES
- ROS : http://wiki.ros.org/melodic/Installation/Ubuntu
- OpenCV : installed with ROS.
- RQT: http://wiki.ros.org/rqt/UserGuide/Install/Groovy
---------------------------------------------------------
# USAGE

## Initial Setup:
  1) Install Dependices listed above
  2) Download repository
  3) Using the terminal, go to the `catkin_ws` folder in the project and enter the following command: `catkin_make`
  4) Run the following commands to install rqt_image_view dependencies:<br />
    `cd src/visualize/`<br />
    `sudo python setup.py install`<br />
  5) Once the dependices are installed run the following commands to allow the startup script to run and open all ROS nodes:    <br />
    `cd ../../` <br />
    `chmod +x startup.sh` <br />
    
  ## Running Detector
   1) Go to root directory of project
   2) Run the following command to run all ROS node: `./startup.sh`<br />
   3) Wait for RQT to open
   4) Once open, paste the image path into the bottom pane and press `enter`
   5) The annotated image will show up in RQT once the the detection is run
   6) Repeate steps 4-5
   7) Once finished run `Ctrl + z` and then `tmux kill-server`
    
 ## Training Detector
  1) Obtain the [MNIST data set](http://yann.lecun.com/exdb/mnist/) in png form.
  2) the folders in the training set must be named after the digit they contain
  2) Open `svm_detector.cpp` in `ocr/catkin_ws/src/visualize/src/`
  3) Change the value of the `TRAINING_PATH` variale to the folder path
  4) Delete `hand_written_detector.yml` in `ocr/catkin_ws/`
  5) Run `catkin_make` from` ocr/catkin_ws/
  6) The run the following Commands: <br />
    `cd ../` <br />
    `./startup.sh` <br />
  7) Wait for the top pane to output `DETECTOR READY`
  8) Detector is ready for use!
  
---------------------------------------------------------
# HOW IT WORKS

Three ROS nodes are being run:
- Detector: Runs the svm detector
- Image: Gets the file path for the image
- rqt_image_view: third party script that outputs image in RQT

The SVM detector is trained by using the [MNIST data set](http://yann.lecun.com/exdb/mnist/). Each image is preprocessed by deskewing, and changing the image to greyscale. The HOG is then calculated and used as the feature vector for the svm.

For detection, once a valid file path is found, the image is the risized if it is too big, and then the contours of the image are found and dialated. The SVM has no way of discriminating between a digit and a non-digit, therefore, the background of the image has to be plain.

Once the contours are found, a bounding box is then used to crop out the contour and apply the same preprocess as the training data. The processed image is then passed to the SVM that will output a prediction. The image will then be annoted and send to RQT for display.

---------------------------------------------------------
# ACKNOWLEDGMENTS:
  - How to deskew and image
    https://docs.opencv.org/4.2.0/dd/d3b/tutorial_py_svm_opencv.html
  
  - How to Contour and image
    https://www.hackevolve.com/recognize-handwritten-digits-1/
    
  - How to convert Vector to Matrix
    https://github.com/ahmetozlu/vehicle_counting_hog_svm/blob/master/src/Main.cpp
    
  - RQT image view
    http://wiki.ros.org/rqt_image_view
