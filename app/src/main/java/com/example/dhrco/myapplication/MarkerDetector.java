package com.example.dhrco.myapplication;

import android.support.annotation.NonNull;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static org.opencv.core.CvType.CV_8UC4;

/**
 * Created by dhrco on 2017-06-13.
 */


public class MarkerDetector {
    final static float MINContourLengthAllowd = 1;
    final static int MINContourPointsAllowed  = 100;
    private static String TAG= "MDetector";
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
            Log.d(TAG+":eps",String.valueOf(eps));
            Imgproc.approxPolyDP(contour, approxCurve, eps, true);

         /* 사각형 검사*/
            Log.d(TAG+":ACSize",String.valueOf(approxCurve.toList().size()));
            if (approxCurve.toList().size() != 4)
                continue;

         /* 볼록 여부 검사*/
            MatOfPoint mat = new MatOfPoint();
            approxCurve.convertTo(mat, CvType.CV_32SC2);
            Log.d(TAG+":isContourC", String.valueOf(Imgproc.isContourConvex(mat)));
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
            Log.d(TAG+":minDist_cmp", String.valueOf(minDist) +" < " + String.valueOf(m_minContourLengthAllowd));
            if (minDist < m_minContourLengthAllowd)
                continue;

         /* 후보 저장 */
            Marker m = new Marker(4, new Vector<Point>());
            Log.d(TAG+":m",m.toString());
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
                Log.d("area", area.toString());
            }
        }
    }

    public void drawMarker(Mat grayscale, Vector<Rect> rects) {
        for (int i = 0; i < rects.size(); i++) {
            Rect area = rects.get(i);
            Log.d(TAG+":area", area.toString());
            Imgproc.rectangle(grayscale, new Point(area.x, area.y), new Point(area.x + area.width, area.y + area.height),new Scalar(255));
            Log.d(TAG+":Rec_start", new Point(area.x, area.y).toString());
            Log.d(TAG+":Rec_end", new Point(area.x + area.width, area.y + area.height).toString());
        }
    }

    public void warpMarkers(Mat img, Vector<Mat> canonicalMarkers, Vector<Marker> detectedMarkers) {
        for (int i = 0; i < detectedMarkers.size(); i++) {
            if (detectedMarkers.get(i).points.size() == 4) {
                Mat canonicalMarker = new Mat(img.height(), img.width(), CV_8UC4);
                Log.d(TAG+":img_height",String.valueOf(img.height()));
                Log.d(TAG+":img_width",String.valueOf(img.width()));
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
                Log.d(TAG+":Mat_M",M.toString());
                Imgproc.warpPerspective(img, canonicalMarker, M, markerSize);
                canonicalMarkers.add(canonicalMarker);
            }
        }
    }

    public Point lowestPoint(List<Point> contour){
        int min = Integer.MAX_VALUE;
        int index_min=-1;

        for (int i = 0; i < contour.size(); i++) {
            if (min > contour.get(i).y) {
                min = (int) contour.get(i).y;
                index_min = i;
            }
        }
        return contour.get(index_min);
    }
    public Point highestPoint(List<Point> contour){
        int max = 0;
        int index_max=-1;

        for (int i = 0; i < contour.size(); i++) {
            if (max < contour.get(i).y) {
                max = (int) contour.get(i).y;
                index_max = i;
            }
        }
        return contour.get(index_max);
    }
    public Point leftPoint(List<Point> contour){
        int max = 0;
        int index_max=-1;

        for (int i = 0; i < contour.size(); i++) {
            if (max < contour.get(i).x) {
                max = (int) contour.get(i).x;
                index_max = i;
            }
        }
        return contour.get(index_max);
    }

    public Point rightPoint(List<Point> contour){
        int min = Integer.MAX_VALUE;
        int index_min= -1;

        for (int i = 0; i < contour.size(); i++) {
            if (min > contour.get(i).x) {
                min = (int) contour.get(i).x;
                index_min = i;
            }
        }

        return contour.get(index_min);
    }

    class XRange extends Range implements Comparable<XRange>{
        XRange(){
            super();
        }
        XRange(int s, int e){
            super(s,e);
        }
        @Override
        public int compareTo(XRange o) {
            if (this.start > o.start)
                return 1;
            if (this.start < o.start)
                return -1;
            else
                return 0;
        }
    }

    public void extractNumbers(Vector<Mat> canonicalMarkers, Vector<Vector<Rect>> rects_numbers ,Vector<Rect> rects){

        rects_numbers.clear();
        for (int i = 0; i < canonicalMarkers.size(); i++) {
            Mat canonicalMarker = canonicalMarkers.get(i);

            Mat numbers = new Mat(canonicalMarker.height(), canonicalMarker.width(), CV_8UC4);
            Imgproc.Canny(canonicalMarker, numbers, 50, 100);

            Vector<MatOfPoint> contours = new Vector<MatOfPoint>();
            mfindContours(numbers, contours, 1);

            Mat contoursfound2 = new Mat(numbers.size(), CvType.CV_8UC4, new Scalar(255, 255, 255));

            double height = canonicalMarker.size().height;
            double ratio2 = height * 0.35;

            Vector<MatOfPoint> figures = new Vector<MatOfPoint>();
            for (int j = 0; j < contours.size(); j++) {
                Vector<MatOfPoint> tmp_contours = new Vector<MatOfPoint>();
                tmp_contours.add(contours.get(j));
                MatOfPoint figure = (MatOfPoint) tmp_contours.get(0);

                if (Imgproc.contourArea(figure) > 10) {
                    if(lowestPoint(figure.toList()).y > ratio2){
                        if(highestPoint(figure.toList()).y < height - ratio2){
                            Imgproc.drawContours(contoursfound2, tmp_contours, -1, new Scalar(0,255,0), 2);;
                            figures.add(figure);
                        }
                    }
                }
            }

            XRange domain = new XRange();
            Vector<XRange> domains = new Vector<XRange>();

            PriorityQueue<XRange> pq = new PriorityQueue<XRange>();

            int numbers_highest= Integer.MAX_VALUE;
            int numbers_lowest = 0;
            for (int j = 0; j < figures.size();j++){
                numbers_highest = Math.min((int) highestPoint(figures.get(j).toList()).y, numbers_highest);
                numbers_lowest = Math.max((int) lowestPoint(figures.get(j).toList()).y, numbers_lowest);
                pq.add(new XRange((int) leftPoint(figures.get(j).toList()).x, (int) rightPoint(figures.get(j).toList()).x));
            }

            XRange own = new XRange();

            while(pq.size()>0){
                own = pq.poll();
                domains.add(own);
            }

            for (int j = 0; j < domains.size(); j++) {
                int flag = domains.size();
                boolean flag2 = false;
                for (int k = j; k > 0; k--) {
                    if (domains.get(j).start >= domains.get(k - 1).end) {
                        flag2 = true;
                        flag = Math.min(k - 1, flag);
                    }
                }
                if (flag2) {
                    domains.get(j).start = domains.get(flag).end;
                    domains.get(j).end = Math.min(domains.get(j).end, domains.get(flag).end);
                    /* j-1 에서 flag 까지 지우기 */
                    for(int k=0;k<j-flag;k++){
                        domains.remove(flag);
                    }
                    j = flag;
                }
            }

            int rects_numbers_x = rects.get(i).x;
            int rects_numbers_y = rects.get(i).y;

            Vector<Rect> rect_number = new Vector<Rect>();
            for (int j = 0; j < domains.size(); j++) {
                rect_number.add(new Rect(new Point(rects_numbers_x+domains.get(j).start, rects_numbers_y+ratio2), new Point(rects_numbers_x+domains.get(j).end, rects_numbers_y+height-ratio2)));
                //Imgproc.rectangle(numbers, new Point(domains.get(j).start, ratio2), new Point(domains.get(j).end, height-ratio2), new Scalar(255), 2);
            }
            rects_numbers.add(rect_number);
        }
    }

    public void cuT(Mat img, Vector<Rect> rects, Vector<Mat> cuts) {
        for(int i=0;i<rects.size();i++){
            Mat cut = new Mat(img.height(), img.width(), CV_8UC4);
            Rect rect = rects.get(i);
            Point[] src = { new Point(rect.x,rect.y),new Point(rect.x+rect.width,rect.y),new Point(rect.x,rect.y+rect.height),new Point(rect.x+rect.width,rect.y+rect.height)};
            Mat M = Imgproc.getPerspectiveTransform(new MatOfPoint2f(src), new MatOfPoint2f(src));
            Imgproc.warpPerspective(img, cut, M, rect.size());
            cuts.add(cut);
        }
    }
}