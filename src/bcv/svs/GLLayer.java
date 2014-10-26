package bcv.svs;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.opengl.Matrix;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.widget.TextView;

public class GLLayer extends GLSurfaceView implements SurfaceHolder.Callback, 
	GLSurfaceView.Renderer, IDisplayFrameEventListener {
	
	private static final String TAG = "GLLayer";
	private final Context mActivityContext;
	// this should always be 0 (black), unless you are interested in debugging
	// opengl sizing, in which case, it may be helpful to set it to faint grey.
	private static final byte backgroundTexColor = (byte)128;
	
	private float texture_size = 1.0f;
	private static int TEX_SIZE_W = 320;
	private static int TEX_SIZE_H = 320;
	
	private static byte[] glCameraFrame = new byte[TEX_SIZE_H*TEX_SIZE_W*3];
	private int[] textureHandle;
	
	private float[] mModelMatrix = new float[16];
	private float[] mViewMatrix = new float[16];

	private float[] mProjectionMatrix = new float[16];
	private float[] mMVPMatrix = new float[16];
	
	private final FloatBuffer mCubePositions;
	private final FloatBuffer mCubeTextureCoordinates;
	
	private int mMVPMatrixHandle;
	private int mTextureUniformHandle;
	private int mPositionHandle;
		
	private int mTextureCoordinateHandle;
	private final int mBytesPerFloat = 4;	
	private final int mPositionDataSize = 3;		
	private final int mTextureCoordinateDataSize = 2;
	private int mProgramHandle;
		
	private TextView fpsTextView = null; // textview with fps readings
	private double prevtime = 0.0;
	private double[] fpsbuffer = new double[10]; 
	private int mPreviewWidth = 1;
	private int mPreviewHeight = 1;
	
	public boolean isFullscreen = false;
	
	private boolean INITIALIZED = false;
	
	public GLLayer(Context activityContext, AttributeSet attrs) {
		super(activityContext, attrs);
		
		this.setEGLContextClientVersion(2);
        this.setRenderer(this);
        this.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        // fill array once
		Arrays.fill(glCameraFrame, (byte)backgroundTexColor);
        Arrays.fill(fpsbuffer, (double)0);
        
		mActivityContext = activityContext;
		// Define points for a cube.		
		// X, Y, Z
		final float[] cubePositionData = {
				// Front face
				(-1.0f)*texture_size, 			texture_size, 	texture_size,				
				(-1.0f)*texture_size, 	(-1.0f)*texture_size, 	texture_size,
						texture_size, 			texture_size, 	texture_size, 
				(-1.0f)*texture_size, 	(-1.0f)*texture_size,	texture_size, 				
						texture_size, 	(-1.0f)*texture_size, 	texture_size,
						texture_size, 			texture_size, 	texture_size
		};
		
		// S, T (or X, Y)
		// Texture coordinate data.
		// Because images have a Y axis pointing downward (values increase as you move down the image) while
		// OpenGL has a Y axis pointing upward, we adjust for that here by flipping the Y axis.
		// What's more is that the texture coordinates are the same for every face.
		final float[] cubeTextureCoordinateData = {												
				// Front face
				0.0f, texture_size, 				
				texture_size, texture_size,
				0.0f, 0.0f,
				texture_size, texture_size,
				texture_size, 0.0f,
				0.0f, 0.0f
		};
		
		// Initialize the buffers.
		mCubePositions = ByteBuffer.allocateDirect(cubePositionData.length * mBytesPerFloat)
        .order(ByteOrder.nativeOrder()).asFloatBuffer();							
		mCubePositions.put(cubePositionData).position(0);		
		
		mCubeTextureCoordinates = ByteBuffer.allocateDirect(cubeTextureCoordinateData.length * mBytesPerFloat)
		.order(ByteOrder.nativeOrder()).asFloatBuffer();
		mCubeTextureCoordinates.put(cubeTextureCoordinateData).position(0);
	}
	
	protected String getVertexShader() {
		return RawResourceReader.readTextFileFromRawResource(mActivityContext, R.raw.vertex_shader);
	}
	
	protected String getFragmentShader() {
		return RawResourceReader.readTextFileFromRawResource(mActivityContext, R.raw.fragment_shader);
	}
	
	public boolean initialize() {
		try {
			System.loadLibrary( "opengl_utils" );
			INITIALIZED = true;
			return true;
		} catch (Exception e) { return false; }
	}
	@Override
	public void onSurfaceCreated(GL10 glUnused, EGLConfig config) {
		// Set the background clear color to black.
		GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		
		// Use culling to remove back faces.
		GLES20.glEnable(GLES20.GL_CULL_FACE);
		
		// Enable depth testing
		GLES20.glEnable(GLES20.GL_DEPTH_TEST);
			
		// Position the eye in front of the origin.
		final float eyeX = 0.0f;
		final float eyeY = 0.0f;
		final float eyeZ = 0.0f;

		// We are looking toward the distance
		final float lookX = 0.0f;
		final float lookY = 0.0f;
		final float lookZ = -10.0f;

		// Set our up vector. This is where our head would be pointing were we holding the camera.
		final float upX = 0.0f;
		final float upY = 1.0f;
		final float upZ = 0.0f;

		// Set the view matrix. This matrix can be said to represent the camera position.
		Matrix.setIdentityM(mViewMatrix, 0);
		Matrix.setLookAtM(mViewMatrix, 0, eyeX, eyeY, eyeZ, lookX, lookY, lookZ, upX, upY, upZ);		

		final String vertexShader = getVertexShader();   		
 		final String fragmentShader = getFragmentShader();			
		
		final int vertexShaderHandle = ShaderHelper.compileShader(GLES20.GL_VERTEX_SHADER, vertexShader);		
		final int fragmentShaderHandle = ShaderHelper.compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentShader);		
		
		mProgramHandle = ShaderHelper.createAndLinkProgram(vertexShaderHandle, fragmentShaderHandle, 
				new String[] {"a_Position", "a_TexCoordinate"});								                                							       
	}	
		
	@Override
	public void onSurfaceChanged(GL10 glUnused, int width, int height) {
		// Set the OpenGL viewport to the same size as the surface.		
		GLES20.glViewport(0, 0, width, height);

		// Create a new perspective projection matrix. The height will stay the same
		// while the width will vary as per aspect ratio.
		final float ratio = ((float) width) / ((float)height);
		final float left = -ratio;
		final float right = ratio;
		final float bottom = -1.0f;
		final float top = 1.0f;
		final float near = 0.5f;
		final float far = 5.0f;
		
		Matrix.setIdentityM(mProjectionMatrix, 0);
		Matrix.orthoM(mProjectionMatrix, 0, left, right, bottom, top, near, far);
		//Matrix.frustumM(mProjectionMatrix, 0, left, right, bottom, top, near, far);
	}	

	@Override
	public void onDrawFrame(GL10 glUnused) {
		GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);			        
                       
        // Set our per-vertex lighting program.
        GLES20.glUseProgram(mProgramHandle);
        
        // Set program handles for cube drawing.
        mMVPMatrixHandle = GLES20.glGetUniformLocation(mProgramHandle, "u_MVPMatrix");
        mTextureUniformHandle = GLES20.glGetUniformLocation(mProgramHandle, "u_Texture");
        mPositionHandle = GLES20.glGetAttribLocation(mProgramHandle, "a_Position");
        mTextureCoordinateHandle = GLES20.glGetAttribLocation(mProgramHandle, "a_TexCoordinate");
        
        // Set the active texture unit to texture unit 0.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        
        // Bind the texture to this unit.
        bindCameraTexture();
        
        // Tell the texture uniform sampler to use this texture in the shader by binding to texture unit 0.
        GLES20.glUniform1i(mTextureUniformHandle, 0);                            
        
        // Draw some cubes.                
        Matrix.setIdentityM(mModelMatrix, 0);
        // fraction
        Matrix.translateM(mModelMatrix, 0, -0.0f, -0.0f, -2.0f);
        // magic scale
        float scale = 1.0f;
        if (isFullscreen) {
        	float s = (16.0f/9.0f) / ((float)mPreviewWidth/(float)mPreviewHeight);
        	scale = ((float)TEX_SIZE_W) / ((float)mPreviewHeight) * s;
        } else {
        	scale = ((float)TEX_SIZE_H) / ((float)mPreviewHeight);
        }
        Matrix.scaleM(mModelMatrix, 0, 1.12f, 1.12f, 1.0f); // magic numbers
        Matrix.scaleM(mModelMatrix, 0, scale, scale, 1.0f);
        Matrix.scaleM(mModelMatrix, 0, 0.5f, 0.5f, 1.0f);
        drawCube();      
	}				
	
	private void drawCube() {		
		// Pass in the position information
		mCubePositions.position(0);		
        GLES20.glVertexAttribPointer(mPositionHandle, mPositionDataSize, GLES20.GL_FLOAT, false,
        		0, mCubePositions);        
                
        GLES20.glEnableVertexAttribArray(mPositionHandle);        
  
        // Pass in the texture coordinate information
        mCubeTextureCoordinates.position(0);
        GLES20.glVertexAttribPointer(mTextureCoordinateHandle, mTextureCoordinateDataSize, GLES20.GL_FLOAT, false, 
        		0, mCubeTextureCoordinates);
        
        GLES20.glEnableVertexAttribArray(mTextureCoordinateHandle);
        
		// This multiplies the view matrix by the model matrix, and stores the result in the MVP matrix
        // (which currently contains model * view).
        Matrix.multiplyMM(mMVPMatrix, 0, mViewMatrix, 0, mModelMatrix, 0);                 
        // This multiplies the modelview matrix by the projection matrix, and stores the result in the MVP matrix
        // (which now contains model * view * projection).
        Matrix.multiplyMM(mMVPMatrix, 0, mProjectionMatrix, 0, mMVPMatrix, 0);
        // Pass in the combined matrix.
        GLES20.glUniformMatrix4fv(mMVPMatrixHandle, 1, false, mMVPMatrix, 0);
        // Draw the cube.
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6);                               
	}	
	
	/**
	 * Generates a texture from the black and white array filled by the on PreviewFrame
	 */
	private void bindCameraTexture(){	
		synchronized(this){
			
			if(textureHandle == null)
				textureHandle = new int[1];
			else
				GLES20.glDeleteTextures(1, textureHandle, 0);
			
			GLES20.glGenTextures(1, textureHandle, 0);

			if(textureHandle[0] != 0 && glCameraFrame != null) {
				GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureHandle[0]);
				GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGB, TEX_SIZE_W, TEX_SIZE_H, 0, 
						GLES20.GL_RGB, GLES20.GL_UNSIGNED_BYTE, ByteBuffer.wrap(glCameraFrame));

				GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
				GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
			}
		}
	}
	public void setFpsTextView(TextView fpstext) {
		fpsTextView = fpstext;
	}
	
	private double calculateFps() {
		long curtime = System.nanoTime();
		double fps = 1000000000.0 / (curtime-prevtime);
		prevtime = curtime;
		//
		double fps_ = 0;
		int N = fpsbuffer.length;
		for (int i = 0; i < (N-1); ++i) {
			fps_ += fpsbuffer[i];
			fpsbuffer[i] = fpsbuffer[i+1];
		}
		fpsbuffer[N-1] = fps;
		fps_ /= (double)(N); 
		return fps_;
	}
	
	private void updateFpsTextView() {
		if (fpsTextView != null) {
			fpsTextView.setText( String.format("fps = %.2f", calculateFps() ));
		}
	}	
	
	// -------------------------------------------------------------------------
	@Override
	public void onEvent(byte[] data, int w, int h, boolean rgb) {
		synchronized (this) {
			//if (!INITIALIZED) { return; } //
			clipframe(data, glCameraFrame, w, h, TEX_SIZE_W, TEX_SIZE_H);
			requestRender();
		}
		updateFpsTextView();
	}
	
	@Override
	public void onPreviewSizeChange(int w, int h) {
		Log.i(TAG, String.format("sup on previewsize change %d %d", w, h) );
		mPreviewWidth = w;
		mPreviewHeight = h;
		int maxval = Math.max(w,h);
		// either resize it up or down..
		if (maxval != Math.max( TEX_SIZE_W, TEX_SIZE_H)) {
			TEX_SIZE_W = maxval;
			TEX_SIZE_H = maxval;			
		}
		// reset texture memory array
		if (glCameraFrame.length != TEX_SIZE_W*TEX_SIZE_H*3) {
			glCameraFrame = new byte[TEX_SIZE_W*TEX_SIZE_H*3];
			Arrays.fill(glCameraFrame, (byte)backgroundTexColor); //
		}
	}
	// -------------------------------------------------------------------------
	    
    private native void clipframe(byte[] in, byte[] out, int width, int height, 
    							  int out_w, int out_h);    

}
