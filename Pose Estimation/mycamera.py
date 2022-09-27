# camera class
class MyCamera(BoxLayout):
    frames_per_second = 60.0
    t = None

    # constructor
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        self.img=Image()
        self.add_widget(self.img)
        self.event = None
        self.texture = None
        self.snackbar = Snackbar(text="",
                                 bg_color=(0.0157, 0.7059, 0.6745, 1),
                                 duration=1)# sth like an alert
        # the video capture object, using webcam
        self.capture = cv2.VideoCapture(0)
        # set the size to the resolution
        self.size = self.get_dims(self.capture)
        # configure the tflite interpreter       
        self.interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_thunder_3.tflite')
        self.interpreter.allocate_tensors()

    # start video recorder
    def start(self):
        # if the video capture is closed
        if not self.capture.isOpened():
            # reopen the video capture webcam
            self.capture.open(0)
            # set the size to the resolution
            self.size = self.get_dims(self.capture)

        # if there’s no event
        if self.event is None:
            # create a new one
            self.event = Clock.schedule_interval(self.update, 1 / self.frames_per_second)
            # start the event
            self.event()
        self.snackbar.text = "Camera started!"
        self.snackbar.open()

    # stop recording
    def stop(self, screenmanager):
        if self.event is not None:
            # cancel the clock event
            self.event.cancel()
            # release the video capture
            self.capture.release()
            self.snackbar.text = "Camera stopped!"
            self.snackbar.open()
        # set the event to `None` again
        self.event = None

    # update the video recorder
    def update(self, *args):
        # read the video capture object
        ret, frame = self.capture.read()
        # copy frame
        img = frame.copy()
        # resize
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
        input_image = tf.cast(img, dtype=tf.float32)
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # make predictions 
        self.interpreter.set_tensor(
                            input_details[0]['index'],
                            np.array(input_image)
                            )
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(
                                        output_details[0]['index'])
        
        # draw the keypoints
        draw_keypoints(frame, keypoints_with_scores, 0.4)
        # flip the frame horizontally
        buf = cv2.flip(frame, 0).tobytes()
        # create texture
        self.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        self.texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        # paste the texture
        self.img.texture = self.texture


    # grab resolution dimensions
    def get_dims(self, cap):
        width, height = (1280, 720)
        self.capture.set(3, width)
        self.capture.set(4, height)
        return width, height
