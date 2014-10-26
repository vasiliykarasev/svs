package bcv.svs;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.ActivityInfo;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.WindowManager.LayoutParams;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.RelativeLayout;

public class MainActivity extends Activity {
	private static final String TAG = "MainActivity";	
	private GLLayer glView;
	private CamLayer mCameraLayer;
	private Processor mProcessor;
	private View mParamsView = null;
	private ImageButton mObjectLearnButton = null;
	private DrawSquare mObjLearnSquare = null;
	private boolean mUiTouchdown = false;
	private Handler mUiHandler = null;	
	
	@Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        
        final Window win = getWindow();
        win.setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        requestWindowFeature(Window.FEATURE_NO_TITLE);

        setContentView(R.layout.activity_main);
        
        Console console = new Console((TextView)findViewById(R.id.main_console) );
        

        glView = (GLLayer)findViewById(R.id.renderer);
        glView.setFpsTextView( (TextView)findViewById(R.id.fpsText) );
        mProcessor = new Processor(this, (IDisplayFrameEventListener)glView, 0, 0);
        mProcessor.setTextView( (TextView)findViewById(R.id.gmmText) );
        mProcessor.addConsole( console );
        mProcessor.gmmLayout = (RelativeLayout)findViewById(R.id.gmm_layout);
        mProcessor.GMMfg = new GMMdata(this, 
        		(RelativeLayout)findViewById(R.id.gmm_fg_data),
        		(TextView)findViewById(R.id.gmm_fg_text));
        
        mProcessor.GMMbg = new GMMdata(this, 
        		(RelativeLayout)findViewById(R.id.gmm_bg_data),
        		(TextView)findViewById(R.id.gmm_bg_text) );    


        mObjLearnSquare = (DrawSquare)findViewById(R.id.drawsquare);        
        mObjectLearnButton = (ImageButton)findViewById(R.id.object_learn_button);
        mObjectLearnButton.setVisibility(View.INVISIBLE);
        
        mObjectLearnButton.setOnTouchListener( new View.OnTouchListener() {
			@Override
			public boolean onTouch(View v, MotionEvent e) {
				if (e.getAction() == MotionEvent.ACTION_DOWN) {
					// post the task.
					mUiTouchdown = true;
					mObjectLearnButton.setImageResource(R.drawable.button_learn2);
					new Thread( new mUiRunnableLearning() ).start();
				    return true;					
				}
				if (e.getAction() == MotionEvent.ACTION_UP) {
					mUiTouchdown = false;
					mObjectLearnButton.setImageResource(R.drawable.button_learn1);					
					mObjectLearnButton.setVisibility(View.INVISIBLE);
					mObjLearnSquare.reset();
					mProcessor.finalizeObjRectangleLearning();
				}
				return false;
			}        	
        });
        // ---------------------------------------------------------------------
        // try to initialize opengl renderer and 'processor' (in other words,
        // try to load all the .so's)
        boolean success = glView.initialize();
        if (!success) {
        	Log.i(TAG, "Having difficulty initializing OpenGL renderer.");
        	android.os.SystemClock.sleep(100); // hang OS for 100ms.
        	success = glView.initialize(); // try again, if this fails, app fails too
        }
        success = mProcessor.initialize();
        if (!success) {
        	Log.i(TAG, "Having difficulty loading libraries...");
        	android.os.SystemClock.sleep(100); // hang OS for 100ms.
        	success = mProcessor.initialize(); // try again, if this fails, app fails too
        }        
        // ---------------------------------------------------------------------        
        
        mCameraLayer = new CamLayer(this, mProcessor);
        LayoutParams lp = new LayoutParams();
        lp.width = 8;
        lp.height = 8;
        addContentView(mCameraLayer, lp);   
        
		mUiHandler = new Handler();
    }
    
	@Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.activity_main, menu);
        return true;
    }
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
	    // Handle item selection
		AlertDialog.Builder builder;
		AlertDialog dialog;
	    switch (item.getItemId()) {
	    
        case R.id.menu_reset: // reset the segmentator
        	if (mProcessor != null) {
	        	mProcessor.pause();
	        	mProcessor.destroyJniObject();
	        	mProcessor.initializeJniObject();
	        	mProcessor.unpause();
	        	Toast.makeText(this, "SSS reset", Toast.LENGTH_SHORT).show();
        	}
            return true;
	    
        case R.id.menu_lock_exposure:
            mCameraLayer.resetExposureLock();
            return true;
        case R.id.menu_res_176x144:
        	mCameraLayer.setResolution(176, 144);
        	return true;
        case R.id.menu_res_320x240:
        	mCameraLayer.setResolution(320, 240);
        	return true;
        case R.id.menu_res_640x480:
        	mCameraLayer.setResolution(640, 480);
        	return true;        	
        case R.id.menu_fullscreen:
        	item.setChecked( !item.isChecked() ); // flip state
        	Log.i("sss", String.format("item checked %b", item.isChecked()) );
        	glView.isFullscreen = item.isChecked();
        	mProcessor.setFullscreen( item.isChecked() );
        	if (item.isChecked()) {
        		String str = "Fullscreen ON.\nNote: a part of the image is now outside of screen boundaries.";
        		Toast.makeText(this, str, Toast.LENGTH_SHORT).show();
        	}
        	return true;
        case R.id.menu_slic:  
        	mParamsView = LayoutInflater.from(this).inflate(R.layout.slic_parameters, null); 
        	((EditText)mParamsView.findViewById(R.id.slic_K)) .setText( Integer.toString(mProcessor.SLIC_K) );
        	((EditText)mParamsView.findViewById(R.id.slic_M)). setText( Integer.toString(mProcessor.SLIC_M) );
        	((EditText)mParamsView.findViewById(R.id.slic_num_iters)). setText( Integer.toString(mProcessor.SLIC_NUM_ITERS) );
        	((CheckBox)mParamsView.findViewById(R.id.slic_show_bdry)). setChecked(mProcessor.SLIC_SHOW_BOUNDARY);
        	((CheckBox)mParamsView.findViewById(R.id.slic_show_graph)). setChecked(mProcessor.SLIC_SHOW_GRAPH);
        	
        	builder = new AlertDialog.Builder(this);
        	builder.setView(mParamsView);
        	builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                @Override
        		public void onClick(DialogInterface dialog, int id) {
                	if (mProcessor != null) {
                		mProcessor.updateSlicParameters(mParamsView);
                		mProcessor.unpause();
                	}
                }
            });
        	builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                @Override
        		public void onClick(DialogInterface dialog, int id) {
                	if (mProcessor != null) { mProcessor.unpause(); }
                }
            });
        	dialog = builder.create();
        	dialog.show();
        	mProcessor.pause();
        	return true;
        case R.id.menu_segmentation:    	
        	mParamsView = LayoutInflater.from(this).inflate(R.layout.segmentation_parameters, null);     
        	
        	((EditText)mParamsView.findViewById(R.id.segmentation_K)) 
        				.setText( Integer.toString(mProcessor.SEGMENTATION_K) );
        	((EditText)mParamsView.findViewById(R.id.segmentation_tv)) 
						.setText( Float.toString(mProcessor.SEGMENTATION_TV_WEIGHT) );
        	((EditText)mParamsView.findViewById(R.id.segmentation_wt)) 
						.setText( Float.toString(mProcessor.SEGMENTATION_WT_WEIGHT) );
        	((EditText)mParamsView.findViewById(R.id.segmentation_beta))
						.setText( Float.toString(mProcessor.SEGMENTATION_BETA) );        	
        	((EditText)mParamsView.findViewById(R.id.segmentation_num_iters))
        				.setText( Integer.toString(mProcessor.SEGMENTATION_NUM_ITERS) );
        	((CheckBox)mParamsView.findViewById(R.id.segmentation_do))
        				.setChecked( mProcessor.SEGMENTATION_ON );
        	((CheckBox)mParamsView.findViewById(R.id.segmentation_show_unary))
						.setChecked( mProcessor.SEGMENTATION_SHOW_UNARY );
        	
        	builder = new AlertDialog.Builder(this);
        	builder.setView(mParamsView);
        	builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                @Override
        		public void onClick(DialogInterface dialog, int id) {
                	if (mProcessor != null) {
                		mProcessor.updateSegmentationParameters(mParamsView);
                		mProcessor.unpause();
                	}
                }
            });
        	builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                @Override
        		public void onClick(DialogInterface dialog, int id) {
                	if (mProcessor != null) {
                		mProcessor.unpause();
                	}
                }
            });
        	dialog = builder.create();
        	dialog.show();
        	mProcessor.pause();
        	return true;
        default:
            return super.onOptionsItemSelected(item);
	    }
	}
	
    public void onGroupItemClick(MenuItem item) {
        // One of the group items (using the onClick attribute) was clicked
        // The item parameter passed here indicates which item it is
        // All other menu item clicks are handled by onOptionsItemSelected()
    }	
	
    @Override
    protected void onSaveInstanceState(Bundle outState) {
       super.onSaveInstanceState(outState);
    }
    
	@Override
    public void onResume(){
		Log.i("sup", "RESUMED");
		super.onResume();
    	glView.onResume();
    	mProcessor.initializeJniObject();
    }
    
	@Override
    public void onPause(){
		Log.i("sup", "PAUSED");
    	super.onPause();
    	glView.onPause();
    	mProcessor.destroyJniObject();
    }
	
	@Override
	public boolean onTouchEvent(MotionEvent e) {
		// do stuff to grow rectangle..
		if(e.getAction() == MotionEvent.ACTION_DOWN) { 
			mObjectLearnButton.setVisibility(View.INVISIBLE);
			mProcessor.initObjRectangle( (int)e.getX(), (int)e.getY() );
	        
			mObjLearnSquare.center_x = (int)e.getX();
	        mObjLearnSquare.center_y = (int)e.getY();
	        mObjLearnSquare.sz = 0;
	        mObjLearnSquare.invalidate();
			mUiTouchdown = true;
		    new Thread( new mUiRunnableGrowRectangle() ).start();   
		    return true;
		}
		else if(e.getAction() == MotionEvent.ACTION_UP) { 
			mUiTouchdown = false;
			boolean created = mProcessor.finalizeObjRectangleGrowth(
					mObjLearnSquare.center_x, mObjLearnSquare.center_y, 
					mObjLearnSquare.sz);
			if (created) { 
				mObjectLearnButton.setVisibility(View.VISIBLE);
			} else {
				mObjLearnSquare.sz = 0;
			}
			// visualize the 'learn' button
			return true;
		}
		return true;
		//return mProcessor.onTouchEvent(e);
	}	
	
    class mUiRunnableGrowRectangle implements Runnable {
        @Override
        public void run() {
        	int sleeptime = 200;
            while (mUiTouchdown) { 
                try { Thread.sleep(sleeptime); }
                catch (InterruptedException e) { e.printStackTrace(); }
                if (!mUiTouchdown) { break; }
                sleeptime = 15;
                mUiHandler.post( 
                	new Runnable() { public void run() {
                		//mProcessor.growObjRectangle();
	                		if (mUiTouchdown){
	                			mObjLearnSquare.sz += 10;                		
	                			mObjLearnSquare.invalidate();
	                		}
                		}
        		});
            }
        }
    }	
    
    class mUiRunnableLearning implements Runnable {
        @Override
        public void run() {
        	int sleeptime = 100;
            while (mUiTouchdown) { 
                try { Thread.sleep(sleeptime); }
                catch (InterruptedException e) { e.printStackTrace(); }
                if (!mUiTouchdown) { break; }
                mUiHandler.post( 
                	new Runnable() { public void run() {
                		mObjLearnSquare.triggerColorFlip();
                		mProcessor.learnObjRectangleTick(); }
        		});
            }
        }
    }    
}
