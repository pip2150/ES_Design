package com.example.dhrco.myapplication;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Vector;

/**
 * Created by dhrco on 2017-06-16.
 */

public class TensorFlowProcesser {

    private static String TAG= "TFProcesser";

    String processing(Classifier classifier, Vector<Mat> canonicalMarkers,Vector<Bitmap> bitmapOutput){
        Bitmap src,bm;
        String floors="";
        String floor = null;

        Log.d(TAG+"CMSize", String.valueOf(canonicalMarkers.size()));

        bitmapOutput.clear();
        for(int j=0;j<canonicalMarkers.size();j++){

            Mat tmp = canonicalMarkers.get(j);
            //tmp = imgGray;

            src = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
//            src = BitmapFactory.decodeResource(getResources(), R.drawable.six);

            /* mat To Bitmap */
            Utils.matToBitmap(tmp, src);
            bitmapOutput.add(src);

            bm = Bitmap.createScaledBitmap(src, 28, 28, true);

            int width = bm.getWidth();
            int height = bm.getHeight();

            // Get 28x28 pixel data from bitmap
            int[] pixels = new int[width * height];
            bm.getPixels(pixels, 0, width, 0, 0, width, height);

            float[] retPixels = new float[pixels.length];
            for (int i = 0; i < pixels.length; ++i) {
                // Set 0 for white and 255 for black pixel
                int pix = pixels[i];
                int b = pix & 0xff;
                //retPixels[i] = 0xff - b;
                retPixels[i] = b;

                if(retPixels[i]>170){
                    retPixels[i] = 0xff;
                }
                else{
                    retPixels[i] = 0;
                }
                pixels[i] = 0x00;
            }
            //bm.setPixels(pixels, 0, width, 0, 0, width, height);

            final List<Classifier.Recognition> results = classifier.recognizeImage(retPixels);
            if (results.size() > 0) {
                floor = results.get(0).getTitle();
                Log.d(TAG+":floor", String.valueOf(floor));
            }

            floors += floor;
            Log.i(TAG+":floors", String.valueOf(floors));
        }

        return floors;
    }
}
