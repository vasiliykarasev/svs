package bcv.svs;

public interface IDisplayFrameEventListener {
	void onEvent(byte[] data, int w, int h, boolean rgb);

	void onPreviewSizeChange(int w, int h);
}
