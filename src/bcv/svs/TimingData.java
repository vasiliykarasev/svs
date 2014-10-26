package bcv.svs;

public class TimingData {
	public double yuv2gray = 0;
	public double yuv2rgb = 0;
	public double gmmlearning = 0;
	public double slic = 0;
	public double slicboundary = 0;
	public double segmentation = 0;
	public double drawcontourboundary = 0;
	TimingData() { }
	
	public void reset() {
		yuv2gray = 0;
		yuv2rgb = 0;
		gmmlearning = 0;
		slic = 0;
		slicboundary = 0;
		segmentation = 0;
		drawcontourboundary = 0;
	}
}
