package com.example.dhrco.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
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
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.RunnableFuture;

import static android.R.attr.bitmap;
import static android.R.attr.delay;
import static java.lang.Math.pow;
import static java.lang.Math.random;
import static java.lang.Math.sqrt;
import static org.opencv.core.CvType.CV_8UC3;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static String TAG= "MainActivity";

    final String serverIP = "isk.iptime.org";
    final int serverPort = 6463;

    private Socket sock;
    private BufferedReader sock_in;
    private PrintWriter sock_out;

    private int iv_id[] = {R.id.imageview0,R.id.imageview1,R.id.imageview2,R.id.imageview3,R.id.imageview4,R.id.imageview5,R.id.imageview6,R.id.imageview7,R.id.imageview8};
    private ImageView iv[] = new ImageView[9];
    private TextView tv;
    private Vector<Bitmap> bitmapOut = new Vector<Bitmap>();
    private String floors="빈칸";

    private static Handler handler;

    private TensorFlowProcesser tensorFlowProcesser;
    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String MODEL_FILE = "file:///android_asset/mnist_model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";

    private JavaCameraView javaCameraView;
    Mat mRgba, imgGray, imgCanny, imgContours;
    MarkerDetector markerDetector;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tensorFlowProcesser = new TensorFlowProcesser();
        handler = new Handler();
        markerDetector = new MarkerDetector();

        new Thread() {
            public void run() {
                try {
                    sock = new Socket(serverIP, serverPort);
                    Log.i(TAG, "Network Connecting!");
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
                        sock_out.println(floors);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }.start();

        for(int i=0;i<iv_id.length;i++)
            iv[i] = (ImageView) findViewById(iv_id[i]);
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
                    classifier = TensorFlowImageClassifier.create(getAssets(), MODEL_FILE, LABEL_FILE, INPUT_SIZE, INPUT_NAME, OUTPUT_NAME);
                    //makeButtonVisible();
                    Log.i(TAG, "Load Success");
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
        mRgba = inputFrame.rgba();
        Log.d(TAG+":CameraSize", String.valueOf(mRgba.size().width)+" * "+String.valueOf(mRgba.size().height));

        /* 그레이스케일 변환 */
        Imgproc.cvtColor(mRgba, imgGray, Imgproc.COLOR_RGB2GRAY);

        /* 영상 이진화 */
        Imgproc.Canny(imgGray, imgCanny, 50, 100);

        /* 윤곽 검출 */
        Vector<MatOfPoint> contours = new Vector<MatOfPoint>();
        markerDetector.mfindContours(imgCanny, contours, MarkerDetector.MINContourPointsAllowed);

        /* 후보 검색 */
        Vector<Marker> detectedMarkers = new Vector<Marker>();
        markerDetector.findCandidates(contours, detectedMarkers);
        Vector<Rect> rects = new Vector<Rect>();
        markerDetector.findRectangle(detectedMarkers, rects);
        markerDetector.drawMarker(imgGray,rects);

        /* 직사각형 2D 변환 */
        Vector<Mat> canonicalMarkers = new Vector<Mat>();
        markerDetector.warpMarkers(imgGray,canonicalMarkers, detectedMarkers);

        /* 관심영역 중 숫자를 Vector<Rect>로 추출 */
        Vector<Vector<Rect>> rects_numbers = new Vector<Vector<Rect>>();
        markerDetector.extractNumbers(canonicalMarkers, rects_numbers, rects);

//        /* Vector<Rect> 그리기 */
//        for(int i=0;i<rects_numbers.size();i++){
//            md.drawMarker(imgGray,rects_numbers.get(i));
//        }

        /* TensorFlow 이미지 처리 */
        floors = tensorFlowProcesser.processing(classifier, canonicalMarkers, bitmapOut);

        handler.post(new Runnable() {
            @Override
            public void run() {
                for(int i=0;i<Math.min(bitmapOut.size(),9);i++){
                    iv[i].setImageBitmap(bitmapOut.get(i));
                }
            }
        });

        return imgGray;
    }
}


