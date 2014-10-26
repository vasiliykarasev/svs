package bcv.svs;

import java.util.LinkedList;
import java.util.ListIterator;

import android.widget.TextView;

public class Console {
	private TextView mTextView = null;
	private LinkedList<String> mData = null;
	private int mMaxlines = 5;

	public Console(TextView sup) {
		mTextView = sup;
		mData = new LinkedList<String>();
		mData.clear();
		updateText();
	}
	
	public void setTextView(TextView sup) { mTextView = sup; }
	
	public void addLine(String s){
		if (mData == null) { return; }
		mData.add(s);
		if (mData.size() > mMaxlines) { mData.pop(); }
		updateText();
	}
	
	private void updateText() {
		if (mTextView == null) { return; }
		String s = "";
		ListIterator<String> it = mData.listIterator();
		while (it.hasNext()) {
			s += it.next();
			s += "\n";
		}
		mTextView.setText(s);
	} 
}
