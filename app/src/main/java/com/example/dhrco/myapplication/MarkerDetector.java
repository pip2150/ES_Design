package com.example.dhrco.myapplication;

import android.support.annotation.NonNull;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Created by dhrco on 2017-06-13.
 */


public class MarkerDetector {
    final static float MINContourLengthAllowd = 1;
    final static int MINContourPointsAllowed  = 100;

    private float m_minContourLengthAllowd = MINContourLengthAllowd;

    public void prepareImage(Mat bgraMat, Mat grayscale) {
        Imgproc.cvtColor(bgraMat, grayscale, Imgproc.COLOR_BGRA2GRAY);
    }

//    public void performThreshold(Mat grayscale, Mat thresholdImg) {
//        adaptiveThreshold(grayscale,
//                thresholdImg,
//                255,
//                ADAPTIVE_THRESH_GAUSSIAN_C,
//                THRESH_BINARY_INV,
//                7,
//                7
//        );
//    }

    public void mfindContours(Mat thresholdImg, Vector<MatOfPoint> contours, int minContourPointsAllowed) {
        Vector<MatOfPoint> allContours = new Vector<MatOfPoint>();

        Imgproc.findContours(thresholdImg, allContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        //findContours(thresholdImg, allContours, RETR_LIST, CHAIN_APPROX_NONE);

        contours.clear();
        for (int i = 0; i < allContours.size(); i++) {
            int contourSize = (int) allContours.get(i).size().area();
            if (contourSize > minContourPointsAllowed)
            {
                contours.add(allContours.get(i));
            }
        }
    }

    public void findCandidates(Vector<MatOfPoint> contours, Vector<Marker> detectedMarkers) {
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        Vector<Marker> possibleMarkers = new Vector<Marker>();

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint2f contour = new MatOfPoint2f();
            contours.get(i).convertTo(contour, CvType.CV_32FC2);

         /* 다각형 근사화*/
            double eps = contours.get(i).size().area() * 0.05;
            Log.d("eps",String.valueOf(eps));
            Imgproc.approxPolyDP(contour, approxCurve, eps, true);

         /* 사각형 검사*/
            Log.d("approxCurve Size",String.valueOf(approxCurve.toList().size()));
            if (approxCurve.toList().size() != 4)
                continue;

         /* 볼록 여부 검사*/
            MatOfPoint mat = new MatOfPoint();
            approxCurve.convertTo(mat, CvType.CV_32SC2);
            Log.d("isContourConvex", String.valueOf(Imgproc.isContourConvex(mat)));
            if (!Imgproc.isContourConvex(mat))
                continue;

         /* 최소 거리 측정 및 비교 */
            //float minDist = numeric_limits<float>::max();
            float minDist = Float.MAX_VALUE;

            for (int j = 0; j < 4; j++)
            {
                Point side = new Point(approxCurve.toArray()[j].x - approxCurve.toArray()[(j + 1) % 4].x, approxCurve.toArray()[j].y - approxCurve.toArray()[(j + 1) % 4].y);
                float squaredSideLegth = (float)side.dot(side);

                minDist = Math.min(minDist, squaredSideLegth);
            }
            Log.d("minDist_cmp", String.valueOf(minDist) +" < " + String.valueOf(m_minContourLengthAllowd));
            if (minDist < m_minContourLengthAllowd)
                continue;

         /* 후보 저장 */
            Marker m = new Marker(4, new Vector<Point>());
            Log.d("m",m.toString());
            for (int j = 0; j < 4; j++)
                m.points.add(new Point(approxCurve.toArray()[j].x, approxCurve.toArray()[j].y));


         /* 후보 시계 방향 정렬*/
            Point v1 = new Point(m.points.get(1).x- m.points.get(0).x,m.points.get(1).y- m.points.get(0).y) ;
            Point v2 = new Point(m.points.get(2).x- m.points.get(0).x,m.points.get(2).y- m.points.get(0).y) ;

            double o = (v1.x * v2.y) - (v1.y * v2.x);

            if (o < 0.0)
                Collections.swap(m.points, 1, 3);

            possibleMarkers.add(m);
        }

        for (int j = 0; j < possibleMarkers.size(); j++) {
            detectedMarkers.add(possibleMarkers.get(j));
        }
    }

    public void findRectangle(Vector<Marker> detectedMarkers, Vector<Rect> rects) {
        for (int i = 0; i < detectedMarkers.size(); i++) {
            if (detectedMarkers.get(i).points.size() == 4) {
                Vector<Point> points = detectedMarkers.get(i).points;
                int x = Math.min((int)points.get(2).x, (int)points.get(3).x);
                int y = Math.min((int)points.get(3).y, (int)points.get(0).y);
                int width = Math.max((int)points.get(0).x, (int)points.get(1).x) - x;
                int height = Math.max((int)points.get(1).y, (int)points.get(2).y) - y;
                Rect area = new Rect(x, y, width, height);
                rects.add(area);
            }
        }
    }

    public void drawMarker(Mat grayscale, Vector<Rect> rects) {
        for (int i = 0; i < rects.size(); i++) {
            Rect area = rects.get(i);
            Imgproc.rectangle(grayscale, new Point(area.x, area.y), new Point(area.x + area.width, area.y + area.height),new Scalar(0, 255, 0));
        }
    }

    public void warpMarkers(Mat img, Vector<Mat> canonicalMarkers, Vector<Marker> detectedMarkers) {
        for (int i = 0; i < detectedMarkers.size(); i++) {
            if (detectedMarkers.get(i).points.size() == 4) {
                Mat canonicalMarker = new Mat(img.height(), img.width(), CvType.CV_8UC4);
                Log.d("img_height",String.valueOf(img.height()));
                Log.d("img_width",String.valueOf(img.width()));
                Marker marker = detectedMarkers.get(i);
                Point[] src = new Point[4];

                for (int j = 0; j < marker.points.size(); j++) {
                    src[j] = marker.points.get(j);
                }
                Log.d("src",src.toString());
                double w1 = sqrt(pow(src[2].x - src[3].x, 2)
                        + pow(src[2].x - src[3].x, 2));
                double w2 = sqrt(pow(src[1].x - src[0].x, 2)
                        + pow(src[1].x - src[0].x, 2));

                double h1 = sqrt(pow(src[1].y - src[2].y, 2)
                        + pow(src[1].y - src[2].y, 2));
                double h2 = sqrt(pow(src[0].y - src[3].y, 2)
                        + pow(src[0].y - src[3].y, 2));

                double maxWidth = (w1 < w2) ? w1 : w2;
                double maxHeight = (h1 < h2) ? h1 : h2;

                Size markerSize = new Size(maxWidth, maxHeight);
                Point m_markerCorners2d[] = { new Point(0,0),
                        new Point(maxWidth - 1, 0),
                        new Point(maxWidth - 1,maxHeight - 1),
                        new Point(0,maxHeight - 1) };

                Mat M = Imgproc.getPerspectiveTransform(new MatOfPoint2f(src), new MatOfPoint2f(m_markerCorners2d));
                Log.d("Mat_M",M.toString());
                Imgproc.warpPerspective(img, canonicalMarker, M, markerSize);
                canonicalMarkers.add(canonicalMarker);
            }
        }
    }


    public void cuT(Mat img, Vector<Rect> rects, Vector<Mat> cuts) {
        for(int i=0;i<rects.size();i++){
            Mat cut = new Mat(img.height(), img.width(), CvType.CV_8UC4);
            Rect rect = rects.get(i);
            Point[] src = { new Point(rect.x,rect.y),new Point(rect.x+rect.width,rect.y),new Point(rect.x,rect.y+rect.height),new Point(rect.x+rect.width,rect.y+rect.height)};
            Mat M = Imgproc.getPerspectiveTransform(new MatOfPoint2f(src), new MatOfPoint2f(src));
            Imgproc.warpPerspective(img, cut, M, rect.size());
            cuts.add(cut);
        }
    }
}