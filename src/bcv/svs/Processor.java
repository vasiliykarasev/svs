package bcv.svs;

import android.content.Context;
import android.graphics.Point;
import android.hardware.Camera;
import android.util.Log;
import android.view.Display;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.CheckBox;
import android.widget.RelativeLayout;
import android.widget.TextView;

public class Processor implements Camera.PreviewCallback {
	private final String TAG = "Processor";
	private final Context mActivityContext;
	private IDisplayFrameEventListener mEventListener;
	private static byte[] mDataRgb = null;
	private static byte[] mDataGray = null;
	private int mPreviewWidth = -1;
	private int mPreviewHeight = -1;
	private int mCamPreviewWidth = -1;
	private int mCamPreviewHeight = -1;
	
	private long mJniObjectAddr = 0L;
	
	public int SLIC_K = 400;
	public int SLIC_M = 2;
	public int SLIC_NUM_ITERS = 1;
	public boolean SLIC_SHOW_BOUNDARY = false;
	public boolean SLIC_SHOW_GRAPH = false;
	public int SEGMENTATION_K = 3; // number of clusters
	public int SEGMENTATION_NUM_ITERS = 50;
	public float SEGMENTATION_BETA = 20.0f;
	public float SEGMENTATION_TV_WEIGHT = 50.0f;
	public float SEGMENTATION_WT_WEIGHT = 1.0f;
	public boolean SEGMENTATION_ON = true;
	public boolean SEGMENTATION_SHOW_UNARY = false;
	
	// must check this with the device on load.
	public int mDeviceWidth = 1920;
	public int mDeviceHeight = 1080;
	// touch event min/max values. they are extremely sensitive and very 
	// dependent to image resolution and opengl disaster in GLlayer.	
	private int mMinY = 0;
	private int mMaxY = mDeviceHeight;
	private int mMinX = 0;
	private int mMaxX = mDeviceWidth;
	private boolean isCameraFullscreen = false;
	
	private boolean PAUSED = false;
	public boolean INITIALIZED = false;

	public GMMdata GMMfg = null;
	public GMMdata GMMbg = null;
	TextView gmmTextView = null;
	public RelativeLayout gmmLayout = null;
	
	public Console mConsole = null; // for now make public.
	public TimingData mTiming = null;
	
	// -------------------------------------------------------------------------
	private boolean mObjRectExists = false;
	private boolean mObjRectSet = false;
	private boolean mObjRectLearning = false;
	private boolean mObjRectLearned = false;	
	private int mObjRectX;
	private int mObjRectY;
	private float mObjRectWidth;
	private float mObjRectHeight;
	
	public Processor(final Context activityContext, IDisplayFrameEventListener listener, int w, int h) {	
		mEventListener = listener;
		mTiming = new TimingData();
		mDataRgb = new byte[w*h*3];
		mDataGray = new byte[w*h];
		mPreviewWidth = w;
		mPreviewHeight = h;
		mActivityContext = activityContext;
		mEventListener.onPreviewSizeChange(mPreviewWidth, mPreviewHeight);
	}	
	public boolean initialize() {
		try {
			System.loadLibrary( "opencv_tools" );
	        System.loadLibrary( "jni_tslic" );
	        System.loadLibrary( "jni_segmentation" );
	        System.loadLibrary( "yuv2rgb" );
	        
	        INITIALIZED = true;        
	        initializeJniObject(); // init object
	        return true;
		} catch (Exception e) {
			return false;
		}
	}
	
	public void onPreviewFrame(byte[] frameByte, Camera camera){
		long t1 = 0;
		synchronized(this) {
			resizeDataArrays( camera.getParameters() );
			// get a copy of rgb data:
		    if (INITIALIZED) {
				t1 = System.nanoTime();
				convertaspectratio(frameByte, mCamPreviewWidth, mCamPreviewHeight, mPreviewWidth, mPreviewHeight);
				yuv2rgb(frameByte, mDataRgb, mPreviewWidth, mPreviewHeight);
				mTiming.yuv2rgb = (double)(System.nanoTime()-t1)/1000000.0;
		    }

			if ((INITIALIZED) && (!PAUSED) ) {
				// -------------------------------------------------------------
				// 						main workhorse
				// -------------------------------------------------------------
				t1 = System.nanoTime();
				nativeRunTSlic(mJniObjectAddr, frameByte); // this is done on grayscale image!!..
				// construct superpixel graph on rgb image
				nativeTSlicConstructGraph(mJniObjectAddr, mDataRgb, 3); 
				mTiming.slic = (double)(System.nanoTime()-t1)/1000000.0;
				if (SLIC_SHOW_BOUNDARY) {
					t1 = System.nanoTime();
					nativeShowTSlicImage(mJniObjectAddr, mDataRgb, 3);
					mTiming.slicboundary = (double)(System.nanoTime()-t1)/1000000.0;
				}
				if (SLIC_SHOW_GRAPH) {
				    drawGraphEdges(mJniObjectAddr, mDataRgb, 3);
				}

				if (mObjRectSet && mObjRectLearning) {
					nativeLearnAddFeatures(mJniObjectAddr, 
							mObjRectX, mObjRectY, (int)mObjRectWidth, (int)mObjRectHeight);
					mObjRectLearning = false;
					
					String str = String.format("model size: bg: %d fg: %d",  
							nativeGetNumPointsFG(mJniObjectAddr), 
							nativeGetNumPointsBG(mJniObjectAddr) );
					mConsole.addLine(str);
				}
				if (mObjRectLearned) {
					t1 = System.nanoTime();
					nativeFinalizeLearning(mJniObjectAddr);
					//nativeLearnGMMfromData(mJniObjectAddr);		
					mTiming.gmmlearning = (double)(System.nanoTime()-t1)/1000000.0;
					getGmmData();
					destroyObjRectangle();
				}
				if ((SEGMENTATION_ON) && (!SEGMENTATION_SHOW_UNARY)) {
					t1 = System.nanoTime();
					doBinarySegmentation(mJniObjectAddr);
					showSegmentationResult(mJniObjectAddr, mDataRgb, mCamPreviewHeight, mCamPreviewWidth, 3);
					mTiming.segmentation = (double)(System.nanoTime()-t1)/1000000.0;
				}
				if (SEGMENTATION_SHOW_UNARY) {
					t1 = System.nanoTime();
					showSegmentationUnary(mJniObjectAddr, mDataRgb, 3);
					mTiming.segmentation = (double)(System.nanoTime()-t1)/1000000.0;
				}
			}			
			String s = String.format("slic: %5.2f, slicvis: %5.2f seg: %5.2f, ",
					mTiming.slic, mTiming.slicboundary, mTiming.segmentation);
			mConsole.addLine(s);
			
			// ideally should have one thread running, and writing camera data
			// into the arrays, ONLY IF these arrays are 'ready' (processing finished)
			
			// at the same time, ONLY if those arrays are ready, can we raise the event for the rendeder.
			
			// ----------------------------------------------------------------
			//                  RAISE AN EVENT FOR OPENGL RENDERER						   
			mEventListener.onEvent( mDataRgb, mPreviewWidth, mPreviewHeight, true );
			// ----------------------------------------------------------------
		}
	}
	
	private void resizeDataArrays(Camera.Parameters p) { 
		int w = p.getPreviewSize().width;
		int h = p.getPreviewSize().height;
		if ((w != mCamPreviewWidth) || (h != mCamPreviewHeight)) { 
			int h_ = ((int)(w * (9.0 / 16.0))/2 ) * 2;
			
			mCamPreviewWidth = w;
			mCamPreviewHeight = h;
			mPreviewWidth = w;
			mPreviewHeight = h_;
			mDataRgb = new byte[w*h_*3];
			mDataGray = new byte[w*h_];
			updateInputBounds();
			// let the GL layer know about the new resolution:
			mEventListener.onPreviewSizeChange(mPreviewWidth, mPreviewHeight);
			initializeJniObject(); // reinitialize, since dimensions changed...
		}
	}
	
	public void setFullscreen(boolean isfullscreen) { 
		isCameraFullscreen = isfullscreen;
		updateInputBounds();
	}
	
	public void updateInputBounds() { 
		// get screen size:
		WindowManager wm = (WindowManager) mActivityContext.getSystemService(Context.WINDOW_SERVICE);
		Display display = wm.getDefaultDisplay();
		Point pt = new Point(0,0);
		display.getSize(pt);
		mDeviceWidth = pt.y;
		mDeviceHeight = pt.x;
		// this cannot be set like this!!!
		int padding = 0;
		if (!isCameraFullscreen) { // maybe this is a bit easier.
			// "vertical" axis (aka short side in portrait mode) is maxed out, so:
			mMinY = 0 + padding;
			mMaxY = mDeviceHeight - padding;
			// "horizontal" axis (aka long side in portrait mode) does not maxout
			// the screen size, and is centered at the middle of the screen.
			// calculate device-pix / screen-pix
			float s = (float)mDeviceHeight / (float)mPreviewHeight;
			mMinX = (int)(0.5f*(mDeviceWidth - mPreviewWidth*s+padding));
			mMaxX = (int)(0.5f*(mDeviceWidth + mPreviewWidth*s-padding));			
		} else {
			// "vertical" axis (aka short side in portrait mode) is maxed out, so:
			mMinX = 0 + padding;
			mMaxX = mDeviceWidth - padding;
			// "horizontal" axis (aka long side in portrait mode) does not maxout
			// the screen size, and is centered at the middle of the screen.
			// calculate device-pix / screen-pix
			float s = (float)mDeviceWidth / (float)mPreviewWidth;
			mMinY = (int)(0.5f*(mDeviceHeight - mPreviewHeight*s+padding));
			mMaxY = (int)(0.5f*(mDeviceHeight + mPreviewHeight*s-padding));						
		}
		String str = String.format("x: %d %d, y: %d %d", mMinX, mMaxX, mMinY, mMaxY);
		mConsole.addLine(str);

		Log.i(TAG, str );
	}
	
	public void updateSlicParameters(View SlicParams) {
		String s;
		boolean needreset = false;
		boolean chk;		
		
		s = ((TextView)SlicParams.findViewById(R.id.slic_K)).getText().toString();
		try { 
			int newval = Integer.parseInt(s);
			if (SLIC_K != newval) { needreset = true; }
			SLIC_K = newval;
		} catch (Exception e) {}
		
		s = ((TextView)SlicParams.findViewById(R.id.slic_M)).getText().toString();
		try { 
			int newval = Integer.parseInt(s);
			if (SLIC_M != newval) { needreset = true; }
			SLIC_M = newval; 
		} catch (Exception e) {}
		
		s = ((TextView)SlicParams.findViewById(R.id.slic_num_iters)).getText().toString();
		try { 
			int newval = Integer.parseInt(s);
			if (SLIC_NUM_ITERS != newval) { needreset = true; }
			SLIC_NUM_ITERS = newval;
		} catch (Exception e) {}
		
		SLIC_SHOW_BOUNDARY = ((CheckBox)SlicParams.findViewById(R.id.slic_show_bdry)).isChecked();
		SLIC_SHOW_GRAPH = ((CheckBox)SlicParams.findViewById(R.id.slic_show_graph)).isChecked();
		
		if (needreset) { initializeJniObject(); }
	}	 	
	
	public void updateSegmentationParameters(View params) {
		String s;
		boolean needreset = false;
		s = ((TextView)params.findViewById(R.id.segmentation_K)).getText().toString();
		try { 
			int newval = Integer.parseInt( s );
			if (SEGMENTATION_K != newval) { needreset = true; }
			SEGMENTATION_K = newval;
		} catch (Exception e) {}
		//
		s = ((TextView)params.findViewById(R.id.segmentation_tv)).getText().toString();
		try { 
			float newval = Float.parseFloat( s );
			if (SEGMENTATION_TV_WEIGHT != newval) { needreset = true; }
			SEGMENTATION_TV_WEIGHT = newval;
		} catch (Exception e) {}
		//
		s = ((TextView)params.findViewById(R.id.segmentation_wt)).getText().toString();
		try { 
			float newval = Float.parseFloat(s);
			if (SEGMENTATION_WT_WEIGHT != newval) { needreset = true; }
			SEGMENTATION_WT_WEIGHT = newval;
		} catch (Exception e) {}
		//		
		s = ((TextView)params.findViewById(R.id.segmentation_num_iters)).getText().toString();
		try { 
			int newval = Integer.parseInt( s );			
			if (SEGMENTATION_NUM_ITERS != newval) { needreset = true; }
			SEGMENTATION_NUM_ITERS = newval;
		} catch (Exception e) {}
		//
		s = ((TextView)params.findViewById(R.id.segmentation_beta)).getText().toString();		
		try { 
			float newval = Float.parseFloat(s);
			if (SEGMENTATION_BETA != newval) { needreset = true; }
			SEGMENTATION_BETA = newval; 
		} catch (Exception e) {}
		//
		SEGMENTATION_ON = ((CheckBox)params.findViewById(R.id.segmentation_do)).isChecked();
		SEGMENTATION_SHOW_UNARY = ((CheckBox)params.findViewById(R.id.segmentation_show_unary)).isChecked();
		
		if (needreset) { initializeJniObject();	}
		return;
	}	
	
	
	public void initializeJniObject() {
		if (!INITIALIZED) { return; } // libraries not yet loaded.
		destroyJniObject();
		mJniObjectAddr = nativeCreateJniObject(
				SLIC_K, SLIC_M, SLIC_NUM_ITERS,
				SEGMENTATION_K, 
				SEGMENTATION_TV_WEIGHT, SEGMENTATION_WT_WEIGHT, 
				SEGMENTATION_BETA, SEGMENTATION_NUM_ITERS, 
				mPreviewHeight, mPreviewWidth, 1);
		if (GMMfg != null) {
			GMMfg.K = SEGMENTATION_K;
			GMMfg.initView();
		}
		if (GMMbg != null) {
			GMMbg.initView();
			GMMbg.K = SEGMENTATION_K;
		}
	}
	public void destroyJniObject() {
		if (!INITIALIZED) { return; } // libraries not yet loaded.
		if (mJniObjectAddr != 0L) {
			nativeDestroyJniObject(mJniObjectAddr);
			mJniObjectAddr = 0L;
		}
		if (GMMfg != null) {
			GMMfg.clearView();
			GMMfg.clearLabel();
		}
		if (GMMbg != null) {
			GMMbg.clearView();
			GMMbg.clearLabel();
		}
	}
	
	public void addConsole(Console c) { mConsole = c; };

	public Point coordsDeviceToImage(int xraw, int yraw) {
		float xrange = (float)(mMaxX - mMinX);
		float yrange = (float)(mMaxY - mMinY);
		int x = (int) ( (0.0f+(xraw - mMinX)/xrange)*mPreviewWidth );
		int y = (int) ( (1.0f-(yraw - mMinY)/yrange)*mPreviewHeight );
		x = Math.min( Math.max(0, x), mPreviewWidth-1 );
		y = Math.min( Math.max(0, y), mPreviewHeight-1 );
		return (new Point(x,y));
	}
		   
	// handle motion events
	public boolean onTouchEvent(MotionEvent e) {
		return true;
	}

	
	public void getGmmData() {
		float[] mu_fg = new float[SEGMENTATION_K*3];
		float[] mu_bg = new float[SEGMENTATION_K*3];
		float[] pi_fg = new float[SEGMENTATION_K];
		float[] pi_bg = new float[SEGMENTATION_K];
		
	    boolean ok = nativeGetGMMdata(mJniObjectAddr, mu_fg, mu_bg, pi_fg, pi_bg);
		if (!ok) { return; }

		if (GMMfg != null) {
			GMMfg.K = SEGMENTATION_K;
			GMMfg.setPi(pi_fg);
			GMMfg.setMu(mu_fg);
			GMMfg.setLabel();
		}
		if (GMMbg != null) {
			GMMbg.K = SEGMENTATION_K;
			GMMbg.setPi(pi_bg);
			GMMbg.setMu(mu_bg);
			GMMbg.setLabel();
		}
	}
	
	public void setTextView(TextView gmmText) { gmmTextView = gmmText; }
	
	public void pause() { PAUSED = true; }
	public void unpause() { PAUSED = false; }
	
	// ------------------------------------------------------------------------
	public void initObjRectangle(int x, int y) {
		mConsole.addLine("initialize object box");
		if (GMMfg != null) { GMMfg.clear(); }
		if (GMMbg != null) { GMMbg.clear(); }

		// flip coordinates for the image.
		Point p = coordsDeviceToImage(y,x);

		mObjRectExists = true;
		mObjRectSet = false;		
		mObjRectX = p.x;
		mObjRectY = p.y;

		mObjRectWidth = 0;
		mObjRectHeight = 0;
		mObjRectLearning = false;
		mObjRectLearned = false;
	}
	
	public void destroyObjRectangle() {
		mConsole.addLine("destroying object box");		
		mObjRectExists = false;
		mObjRectSet = false;
		mObjRectLearning = false;
		mObjRectLearned = false;		
	}
	
	public boolean finalizeObjRectangleGrowth(int x, int y, int sz) {
		Point p = coordsDeviceToImage(y,x);
		mObjRectX = p.x;
		mObjRectY = p.y;
		mObjRectWidth  = (sz/(float)(mDeviceHeight))*((float)mPreviewHeight);
		mObjRectHeight = (sz/(float)(mDeviceWidth))*((float)mPreviewWidth);		
		
		mConsole.addLine("finalized object box growing");				
		if ((mObjRectHeight < 10) || (mObjRectWidth < 10)) {
			mObjRectSet = false;
			destroyObjRectangle();
			return false;
		} else {
			mObjRectSet = true;
			return true;
		}
	}
	public boolean isObjRectangleSet() { 
		mConsole.addLine( String.format("set = %b", mObjRectSet) );
		return mObjRectSet; 
	}
	
	public void learnObjRectangleTick() {		
		mObjRectLearning = true;
		mConsole.addLine("learning object");
	};
	
	public void finalizeObjRectangleLearning() {
		mConsole.addLine("finalized object learning");
		mObjRectLearning = false;
		mObjRectLearned = true;
	};
	
	
	public boolean isInsideLearningRectangleBounds(int x, int y) {
		if (!mObjRectExists || !mObjRectSet) { return false; }
		Point p = coordsDeviceToImage(y,x);
		boolean inx =  ( (p.x >= mObjRectX - 0.5*mObjRectWidth) && 
					     (p.x <= mObjRectX + 0.5*mObjRectWidth) );
		boolean iny =  ( (p.y >= mObjRectY - 0.5*mObjRectHeight) && 
			     		 (p.y <= mObjRectY + 0.5*mObjRectHeight) );		
		return (inx & iny);
	}
	
	// ------------------------------------------------------------------------
	//						only the native hooks follow
	// ------------------------------------------------------------------------
    
    private native void yuv420rgb(byte[] in, int width, int height, 
    							  byte[] out, int out_w, int out_h);
    private native void yuv2rgb(byte[] in, byte[] out, int width, int height);
    private native void y2gray(byte[] in, byte[] out, int width, int height);
    private native void convertaspectratio(byte[] inout, int w, int h, int outw, int outh);
    
    
    private native long nativeCreateJniObject(
    		int slic_K, int slic_M, int slic_num_iters, int seg_K, 
    		float seg_TV, float seg_wt, float seg_beta, int seg_num_iters,
    		int rows, int cols, int chan);
    private native void nativeDestroyJniObject(long addr);
    
    private native void nativeRunTSlic(long addr, byte[] data);        
    private native void nativeShowTSlicImage(long addr, byte[] data, int chan);
    private native void nativeTSlicConstructGraph(long addr, byte[] data, int chan);
    
    private native void drawContourBoundary(byte[] data, int rows, int cols, int chan, int[] x, int[] y, int n, boolean finished);
    
    private native void nativeSetContourData(long addr, int[] x, int [] y, int n);
	
    // add features from the current image - from superpixels within bounding box
    private native void nativeLearnAddFeatures(long addr, int x, int y, int w, int h);
    // iteratively estimate GMMs
    private native void nativeFinalizeLearning(long addr);
    
	//! get parameters of learned GMMs
    private native boolean nativeGetGMMdata(long addr, 
            float[] mu_fg, float[] mu_bg, float[] pi_fg, float[] pi_bg);
    
    // learn GMM models, given the data stored in the JNI struct.
    private native void nativeLearnGMMfromData(long addr);
    
    private native void doBinarySegmentation(long addr);
    
    private native void showSegmentationResult(long addr, byte[] rgbimg, int rows, int cols, int chan);
    
    private native void showSegmentationUnary(long addr, byte[] img, int chan);
    
    private native void drawGraphEdges(long addr, byte[] data, int chan);
    
    // return number of points used to learn GMMs
    private native int nativeGetNumPointsFG(long addr);
    private native int nativeGetNumPointsBG(long addr);
}