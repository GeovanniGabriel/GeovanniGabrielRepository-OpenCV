package com.example.miprimeropencv

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import org.opencv.imgproc.Imgproc
import org.opencv.android.CameraActivity
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc.resize
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : CameraActivity(), CvCameraViewListener2 {

    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private lateinit var imageMat: Mat
    private lateinit var grayMat: Mat
    var faceDetector: CascadeClassifier? = null
    lateinit var faceDir: File
    private var imageRatio = 0.0

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")

                    loadFaceLib()

                    if (faceDetector!!.empty()) {
                        faceDetector = null
                    } else {
                        faceDir.delete()
                    }

                    mOpenCvCameraView!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        mOpenCvCameraView = findViewById<View>(R.id.myCamera) as CameraBridgeViewBase
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCameraIndex(CameraCharacteristics.LENS_FACING_FRONT)
        mOpenCvCameraView!!.setCvCameraViewListener(this)
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        imageMat = Mat(width, height, CvType.CV_8UC4)
        grayMat = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        imageMat.release()
        grayMat.release()
    }

    override fun getCameraViewList(): List<CameraBridgeViewBase> {
        return listOf(mOpenCvCameraView!!)
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        imageMat = inputFrame.rgba()
        grayMat = get480Image(inputFrame.gray())
        drawFaceRectangle()
        return imageMat
    }

    private fun ratioTo480(src: Size): Double {
        val w = src.width
        val h = src.height
        val heightMax = 480
        var ratio: Double = 0.0

        if (w > h) {
            if (w < heightMax) return 1.0
            ratio = heightMax / w
        } else {
            if (h < heightMax) return 1.0
            ratio = heightMax / h
        }

        return ratio
    }

    private fun get480Image(src: Mat): Mat {
        val imageSize = Size(src.width().toDouble(), src.height().toDouble())
        imageRatio = ratioTo480(imageSize)

        if (imageRatio.equals(1.0)) return src

        val dstSize = Size(imageSize.width * imageRatio, imageSize.height * imageRatio)
        val dst = Mat()
        resize(src, dst, dstSize)
        return dst
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView?.let { mOpenCvCameraView!!.disableView() }
        if (faceDir.exists()) faceDir.delete()
    }

    override fun onPause() {
        super.onPause()
        mOpenCvCameraView?.let { mOpenCvCameraView!!.disableView() }
    }

    fun loadFaceLib() {
        try {
            val modelInputStream = resources.openRawResource(R.raw.haarcascade_frontalface_alt2)

            // create a temp directory
            faceDir = getDir(FACE_DIR, Context.MODE_PRIVATE)

            // create a model file
            val faceModel = File(faceDir, FACE_MODEL)

            // copy model to new face library
            val modelOutputStream = FileOutputStream(faceModel)

            val buffer = ByteArray(byteSize)
            var byteRead = modelInputStream.read(buffer)
            while (byteRead != -1) {
                modelOutputStream.write(buffer, 0, byteRead)
                byteRead = modelInputStream.read(buffer)
            }

            modelInputStream.close()
            modelOutputStream.close()

            faceDetector = CascadeClassifier(faceModel.absolutePath)
        } catch (e: IOException) {
            Log.e(TAG, "Error loading cascade face model...$e")
        }

    }

    private fun drawFaceRectangle() {
        val faceRects = MatOfRect()

        faceDetector!!.detectMultiScale(
            grayMat,
            faceRects
        )

        for (rect in faceRects.toArray()) {
            var x = 0.0
            var y = 0.0
            var w = 0.0
            var h = 0.0

            if (imageRatio.equals(1.0)) {
                x = rect.x.toDouble()
                y = rect.y.toDouble()
                w = x + rect.width
                h = y + rect.height
            } else {
                x = rect.x.toDouble() / imageRatio
                y = rect.y.toDouble() / imageRatio
                w = x + (rect.width / imageRatio)
                h = y + (rect.height / imageRatio)
            }

            Imgproc.rectangle(
                imageMat,
                Point(x, y),
                Point(w, h),
                Scalar(255.0, 0.0, 0.0)
            )
        }
    }

    companion object {
        private const val TAG = "OCVSample::Activity"
        private const val FACE_DIR = "facelib"
        private const val FACE_MODEL = "haarcascade_frontalface_alt2.xml"
        private const val byteSize = 4096
    }
}