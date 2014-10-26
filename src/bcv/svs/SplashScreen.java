package bcv.svs;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;


public class SplashScreen extends Activity {

	private static int TIMEOUT = 1000;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		Log.i("sup", "SUP DOG");
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_splash);

		new Handler().postDelayed(new Runnable() {
			@Override
			public void run() {
				// This method will be executed once the timer is over
				// Start your app main activity
				Intent i = new Intent(SplashScreen.this, MainActivity.class);
				startActivity(i);
				// close this activity
				finish();
			}
		}, TIMEOUT);
	}	
}
