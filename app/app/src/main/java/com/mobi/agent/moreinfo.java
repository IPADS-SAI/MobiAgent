package com.mobi.agent;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import com.google.android.material.appbar.MaterialToolbar;

public class moreinfo extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_moreinfo);
        
        // 设置 Toolbar 返回按钮
        MaterialToolbar toolbar = findViewById(R.id.toolbar);
        toolbar.setNavigationOnClickListener(v -> {
            finish();
            // 添加过渡动画
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
        });
    }
}