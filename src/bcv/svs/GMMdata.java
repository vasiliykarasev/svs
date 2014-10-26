package bcv.svs;

import android.content.Context;
import android.graphics.Color;
import android.util.Log;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.TextView;

public class GMMdata {
	private final String TAG = "GMMdata";
	public float[] mu = null;
	public float[] pi = null;
	public float[] cov = null;
	private final int TOTAL_WIDTH = 800;
	private final int DIM = 3;
	public int K = 0;
	public int dim = 0;
	public Context mContext = null;
	public RelativeLayout mGmmLayout = null;
	public TextView mTextLabel = null;
	private int[] viewid = null;

	GMMdata(Context c, RelativeLayout rlColors, TextView tvlabel) {
		mContext = c;
		mGmmLayout = rlColors;
		mTextLabel = tvlabel;
	}
	
	public void initView() {		
        viewid = new int[K];
		int prev_id = 0;
		int id;
        for (int k = 0; k < K; ++k) { 
            View v = new View(mContext);
            v.setBackgroundColor(Color.BLACK); // Color.argb(a,r,g,b).
            id = getFreeId();
            v.setId(id);
	        RelativeLayout.LayoutParams rlp = new RelativeLayout.LayoutParams(
	        				RelativeLayout.LayoutParams.MATCH_PARENT, 
	        				RelativeLayout.LayoutParams.MATCH_PARENT);
			rlp.height = 35;
			rlp.width = 0; // this initially makes the bars invisible.
			rlp.rightMargin = 5;
			rlp.topMargin = 5;
			rlp.bottomMargin = 5;
			rlp.leftMargin = 5;
			
			if (k == 0) {
				rlp.addRule(RelativeLayout.ALIGN_PARENT_LEFT);
			} else {
				rlp.addRule(RelativeLayout.RIGHT_OF, prev_id);
			}
	        v.setLayoutParams(rlp);
	        mGmmLayout.addView(v);
	        viewid[k] = id;
	        prev_id = id;
        }
	}
	
	public void clearView() { mGmmLayout.removeAllViews(); }
	
	private int getFreeId() {
        int id;
		while (true) {
        	id = (int)Math.floor(Math.random()*100000);
        	if (mGmmLayout.findViewById(id) == null) { break; }
        }
        return id;
	}
	
	public void setPi(float[] data) {
		pi = data;
		for (int k = 0; k < K; ++k) {
			View v = (View)mGmmLayout.findViewById(viewid[k]);
			v.getLayoutParams().width = (int)(data[k]*TOTAL_WIDTH);
		}
	}
	
	public void setMu(float[] mu) {
		if (viewid.length != K) { Log.e(TAG, "views not initialized.."); }
		for (int k = 0; k < K; ++k) {
			float r = mu[DIM*k];
			float g = mu[DIM*k+1];
			float b = mu[DIM*k+2];
			View v = (View)mGmmLayout.findViewById(viewid[k]);
			int c = Color.argb(255, (int)(r*255), (int)(g*255), (int)(255*b) );
			v.setBackgroundColor( c );
		}
	}
	public void clear() {
		clearLabel();
		int num = viewid.length;
		for (int k = 0; k < num; ++k) {
			View v = (View)mGmmLayout.findViewById(viewid[k]);
			v.getLayoutParams().width = 0;
		}
	}
	
	public void clearLabel() { mTextLabel.setText(""); }
	public void setLabel() { mTextLabel.setText("GMM: "); }
	
}
