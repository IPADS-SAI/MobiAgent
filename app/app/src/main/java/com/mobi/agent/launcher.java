package com.mobi.agent;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import com.google.android.material.button.MaterialButton;

import androidx.appcompat.app.AppCompatActivity;

public class launcher extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_launcher);
        
        // 启动按钮
        MaterialButton launchButton = findViewById(R.id.launch_button);
        launchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(launcher.this, MainActivity.class);
                startActivity(intent);
                // 添加过渡动画
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
            }
        });
        
        // 详情按钮
        MaterialButton moreInfoButton = findViewById(R.id.more_info);
        moreInfoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(launcher.this, moreinfo.class);
                startActivity(intent);
                // 添加过渡动画
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
            }
        });
    }
}