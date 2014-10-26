package bcv.svs;

public class PointList {
	public int[] x = null;
	public int[] y = null;
	public int n = 0;
	public int cap = 0;
	
	PointList(int m) {
		cap = m;
		n = 0;
		x = new int[m];
		y = new int[m];
	}
	public void reset() {
		n = 0;
	}
	public void push(int x_, int y_) {
		if (n >= cap) { return; } //screw it
		x[n] = x_;
		y[n] = y_;
		n++;
	}
}
