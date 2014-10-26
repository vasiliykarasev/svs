package bcv.svs;

import java.io.IOException;

import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.Toast;


public class CamLayer extends SurfaceView implements SurfaceHolder.Callback, PreviewCallback {
	private final String TAG = "CamLayer";
	public static int nativeWidth = 176;
	public static int nativeHeight = 144;
    Camera mCamera = null;
    Context mContext;
    boolean isPreviewRunning = false;
    Camera.PreviewCallback callback;
    boolean mNeedExposureLocked = false;
    long mNeedExposureTimeLocked = 0L;
    @SuppressWarnings("deprecation")
	CamLayer(Context context, Camera.PreviewCallback callback) {
        super(context);
        mContext = context;
        this.callback=callback;
        
        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        SurfaceHolder mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    public void surfaceCreated(SurfaceHolder holder) {
    	synchronized(this) {
	        mCamera = Camera.open(0);
	
	    	Camera.Parameters p = mCamera.getParameters();  
	    	int[] range = new int[2];
	    	p.getPreviewFpsRange(range);
	    	Log.i(TAG, String.format("preview range: %d %d", range[0], range[1]));
	    	p.setPreviewFpsRange(4000, 15000);
	    	p.setPreviewSize(nativeWidth, nativeHeight);
	    	mCamera.setDisplayOrientation(0);
	    	mCamera.setParameters(p);
	    	
	    	//do not display the camera preview..!!
	    	try {
				mCamera.setPreviewDisplay(holder);
			} catch (IOException e) {
				Log.e("Camera", "mCamera.setPreviewDisplay(holder);");
			}

	    	mCamera.startPreview();
    		//mCamera.setPreviewCallback(this);
    		mCamera.setPreviewCallbackWithBuffer(this);
	    	byte[] buffer = new byte[nativeWidth*nativeWidth*20];
            mCamera.addCallbackBuffer(buffer);
    	}
	}

    public void surfaceDestroyed(SurfaceHolder holder) {
        // Surface will be destroyed when we return, so stop the preview.
        // Because the CameraDevice object is not a shared resource, it's very
        // important to release it when the activity is paused.
    	synchronized(this) {
	    	try {
		    	if (mCamera!=null) {
		    		mCamera.stopPreview();
		    		mCamera.setPreviewCallback(null);
		    		isPreviewRunning=false;
		    		mCamera.release();
		    	}
	    	} catch (Exception e) {
				Log.e("Camera", e.getMessage());
	    	}
    	}
    }

	// Called when holder has changed
	public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
		synchronized(this) { mCamera.startPreview(); }
	}

	public void onPreviewFrame(byte[] arg0, Camera arg1) {
		if ((mNeedExposureLocked) && 
				(System.currentTimeMillis() > mNeedExposureTimeLocked)) {
			lockExposure();
		}
		synchronized(this) {
			if (callback!=null)
	    		callback.onPreviewFrame(arg0, arg1);
			mCamera.setPreviewCallbackWithBuffer(this);
			mCamera.addCallbackBuffer(arg0); // keep the same buffer.
		}
	}

	public void resetExposureLock() {
		if (mCamera == null) { return; }
		Camera.Parameters p = mCamera.getParameters();
		p.setAutoExposureLock( false );
		mCamera.setParameters(p);
		mNeedExposureLocked = true;
		mNeedExposureTimeLocked = System.currentTimeMillis()+1000;
		Toast.makeText(mContext, "Resetting...", Toast.LENGTH_SHORT).show();
	}
	private void lockExposure() {
		Camera.Parameters p = mCamera.getParameters();
		p.setAutoExposureLock( true );
		mCamera.setParameters(p);
		mNeedExposureLocked = false;
		mNeedExposureTimeLocked = 0;
		Toast.makeText(mContext, "Exposure reset", Toast.LENGTH_SHORT).show();
	}
	public void setResolution(int w, int h) { 
		if (mCamera == null) { return; }
		synchronized (this) {
			mCamera.stopPreview();
	    	Camera.Parameters p = mCamera.getParameters();  
	    	p.setPreviewSize(w,h);
	    	mCamera.setDisplayOrientation(0);
	    	mCamera.setParameters(p);
	    	byte[] buffer = new byte[w*h*4];
            mCamera.addCallbackBuffer(buffer);
	    	nativeWidth = w;
	    	nativeHeight = h;
	    	// crap output 9x16 ends up being difficult to say the least.
	    	//outWidth = w;
	    	//outHeight = h; //(int)Math.round( ((double)w)*(9.0/16.0) );
	    	
    		mCamera.setPreviewCallbackWithBuffer(this);
            mCamera.startPreview();
		}
	}
}