package bcv.svs;
import java.io.IOException;

import bcv.svs.MainActivity.mUiRunnableGrowRectangle;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Handler;
import android.util.AttributeSet;
import android.view.View;

public class DrawSquare extends View {
    Paint paint = new Paint();
    public int center_x = 0;
    public int center_y = 0;
    public int sz = 0;
    private int COLOR_BASE = Color.WHITE;
    private int COLOR_CHG = Color.BLACK;
    public DrawSquare(Context context, AttributeSet attrs) {
        super(context, attrs);
    	COLOR_CHG = Color.parseColor("#348F50"); //R.color.MistBlue; //R.color.blue1;
    	COLOR_BASE = Color.rgb( 86, 180, 211 ); //.parseColor("#56B4D3"); //R.color.MarbleBlue; //green1;
        paint.setColor(COLOR_BASE);
        paint.setStrokeWidth(10);
    }

    private void setColor(int c) {
    	paint.setColor(c);
    	this.invalidate();
    }
    @Override
    public void onDraw(Canvas canvas) {
            canvas.drawLine(center_x-0.5f*sz, center_y-0.5f*sz, center_x+0.5f*sz, center_y-0.5f*sz, paint);
            canvas.drawLine(center_x+0.5f*sz, center_y-0.5f*sz, center_x+0.5f*sz, center_y+0.5f*sz, paint);
            canvas.drawLine(center_x+0.5f*sz, center_y+0.5f*sz, center_x-0.5f*sz, center_y+0.5f*sz, paint);
            canvas.drawLine(center_x-0.5f*sz, center_y-0.5f*sz, center_x-0.5f*sz, center_y+0.5f*sz, paint);
    }

	public void triggerColorFlip() {
		if (paint.getColor() == COLOR_BASE) { setColor(COLOR_CHG); } 
		else { setColor(COLOR_BASE); }
	}
	public void reset() {
		sz = 0;
		center_x = 0;
		center_y = 0;
		setColor(COLOR_BASE);
	}
}
