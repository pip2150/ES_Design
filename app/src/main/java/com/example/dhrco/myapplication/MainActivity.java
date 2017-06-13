package com.example.dhrco.myapplication;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Vector;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }

    JavaCameraView javaCameraView;
    private static String TAG= "MainActivity";
    Mat mRgba, imgGray, imgCanny;
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

        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

//        // Example of a call to a native method
//        TextView tv = (TextView) findViewById(R.id.sample_text);
//        tv.setText(stringFromJNI());
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
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        MarkerDetector md = new MarkerDetector();

        mRgba = inputFrame.rgba();

        /* Grayscale 변환 */
        Imgproc.cvtColor(mRgba, imgGray, Imgproc.COLOR_RGB2GRAY);

        /* 윤관석 도출 */
        Imgproc.Canny(imgGray, imgCanny, 50, 100);

        Vector<MatOfPoint> contours = new Vector<MatOfPoint>();
        md.mfindContours(imgCanny, contours, MarkerDetector.MINContourPointsAllowed);

        Vector<Marker> detectedMarkers = new Vector<Marker>();
        md.findCandidates(contours, detectedMarkers);
        Mat contoursfound = new Mat(imgCanny.size(), 0, new Scalar(255));
        Imgproc.drawContours(contoursfound, contours, -1, new Scalar(1), 2);

        Vector<MatOfPoint> detectedPoints = new Vector<MatOfPoint>();
        for (int i = 0; i < detectedMarkers.size(); i++) {
            MatOfPoint tmp = new MatOfPoint();
            for (int j = 0; j < detectedMarkers.get(i).points.size().area(); j++) {
                tmp.push_back(new MatOfPoint(new Point(detectedMarkers.get(i).points.toArray()[j].x, detectedMarkers.get(i).points.toArray()[j].y)));
            }
            detectedPoints.add(tmp);
        }
        for (int i = 0; i < detectedPoints.size(); i++) {
            if (detectedPoints.get(i).size().area() == 4) {
                MatOfPoint points = detectedPoints.get(i);
                int x = Math.min((int)points.toArray()[2].x, (int)points.toArray()[3].x);
                int y = Math.min((int)points.toArray()[3].y, (int)points.toArray()[0].y);
                int width = Math.max((int)points.toArray()[0].x, (int)points.toArray()[1].x) - x;
                int height = Math.max((int)points.toArray()[1].y, (int)points.toArray()[2].y) - y;
                Rect area = new Rect(x, y, width, height);
                Imgproc.rectangle(imgGray, new Point(x,y), new Point(x+width,y+height), new Scalar(255));
//                Imgproc.rectangle(imgGray, area, new Scalar(255), 2);
            }
        }

        return contoursfound;
    }
}


