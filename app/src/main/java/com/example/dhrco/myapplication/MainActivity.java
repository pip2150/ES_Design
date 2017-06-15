package com.example.dhrco.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    final String serverIP = "isk.iptime.org";
    final int serverPort = 6463;
    Socket sock;
    BufferedReader sock_in;
    PrintWriter sock_out;

    ImageView iv;
    Bitmap src;
    Bitmap bm;
    TextView tv;

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/mnist_model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();
    static String floor="";

    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }

    JavaCameraView javaCameraView;
    private static String TAG= "MainActivity";
    Mat mRgba, imgGray, imgCanny, imgContours;
    BaseLoaderCallback mloaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status){
            switch(status){
                case BaseLoaderCallback.SUCCESS:{
                    javaCameraView.enableView();
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }

        }
    };

    static{
        if(OpenCVLoader.initDebug()){
            Log.i(TAG, "Opencv loaed successfully");
        }
        else{
            Log.i(TAG, "Opencv not loaded");
        }
    }

//    static{
//        if(){
//
//        }
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Thread PrepareSoc = new Thread() {
            public void run() {
                try {
                    sock = new Socket(serverIP, serverPort);
                    Log.i("insu", "connecting!");
                    sock_out = new PrintWriter(sock.getOutputStream(), true);
                    sock_in = new BufferedReader(new InputStreamReader(sock.getInputStream()));
                    sock_out.println("camera");

                    while(true){
                        try{
                            Thread.sleep(1000);
                        }
                        catch(Exception e){
                            e.printStackTrace();
                        }
                        sock_out.println(floor);

                    }
                } catch (IOException e) {
                    Log.i("insu", e.toString());
                    e.printStackTrace();
                }
            }
        };
        PrepareSoc.start();
        iv = (ImageView) findViewById(R.id.imageview);
        tv = (TextView) findViewById(R.id.textview);
        initTensorFlowAndLoadModel();

        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    //makeButtonVisible();
                    Log.i("insu", "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onResume(){
        super.onResume();
        if(OpenCVLoader.initDebug()){
            Log.i(TAG, "Opencv loaed successfully");
            mloaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            Log.i(TAG, "Opencv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mloaderCallback);
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    //public native String stringFromJNI();

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        imgGray = new Mat(height, width, CvType.CV_8UC4);
        imgCanny = new Mat(height, width, CvType.CV_8UC4);
        imgContours = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        MarkerDetector md = new MarkerDetector();

        mRgba = inputFrame.rgba();

        /* 그레이스케일 변환 */
        Imgproc.cvtColor(mRgba, imgGray, Imgproc.COLOR_RGB2GRAY);

        /* 영상 이진화 */
        Imgproc.Canny(imgGray, imgCanny, 50, 100);

        /* 윤곽 검출 */
        Vector<MatOfPoint> contours = new Vector<MatOfPoint>();
        md.mfindContours(imgCanny, contours, MarkerDetector.MINContourPointsAllowed);

        /* 후보 검색 */
        Vector<Marker> detectedMarkers = new Vector<Marker>();
        md.findCandidates(contours, detectedMarkers);
        Vector<Rect> rects = new Vector<Rect>();
        md.findRectangle(detectedMarkers, rects);
        md.drawMarker(imgGray,rects);

        Vector<Mat> canonicalMarkers = new Vector<Mat>();
        md.warpMarkers(imgGray,canonicalMarkers, detectedMarkers);

        Log.i("insu", "where0");
        Mat tmp = null;
        if(canonicalMarkers.size()>-1){
            //tmp = canonicalMarkers.get(0);
            tmp = imgGray;

            Log.i("insu", "where1");
            src = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
//            src = BitmapFactory.decodeResource(getResources(), R.drawable.six);
            Log.i("insu", "where2");
        /* mat To Bitmap */
            Utils.matToBitmap(tmp, src);
            bm = Bitmap.createScaledBitmap(src, 28, 28, true);

            int width = bm.getWidth();
            int height = bm.getHeight();

            // Get 28x28 pixel data from bitmap
            int[] pixels = new int[width * height];
            bm.getPixels(pixels, 0, width, 0, 0, width, height);
            Log.i("insu", "where3");
            float[] retPixels = new float[pixels.length];
            for (int i = 0; i < pixels.length; ++i) {
                // Set 0 for white and 255 for black pixel
                int pix = pixels[i];
                int b = pix & 0xff;
//                retPixels[i] = 0xff - b;
                retPixels[i] = b;

                if(retPixels[i]>170){
                    retPixels[i] = 0xff;
                }else{
                    retPixels[i] = 0;
                }
                pixels[i] = 0x00;
            }
//            bm.setPixels(pixels, 0, width, 0, 0, width, height);
            Log.i("insu", "where4");
            final List<Classifier.Recognition> results = classifier.recognizeImage(retPixels);
            if (results.size() > 0) {
                floor = ""+results.get(0).getTitle();
            }
            Log.i("insu", "where5");
            new Thread(new Runnable()
            {
                @Override
                public void run()
                {
                    runOnUiThread(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            //iv.setImageBitmap(src);
                            tv.setText("Number : "+floor);
                        }
                    });
                }
            }).start();
        }
        return imgGray;
    }
}


