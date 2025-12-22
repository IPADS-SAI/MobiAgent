package com.mobi.agent;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class MessageAdapter extends RecyclerView.Adapter<MessageAdapter.MyViewHolder> {

    List<Message> messageList;

    public MessageAdapter(List<Message> messageList) {
        this.messageList = messageList;
    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View chatView = LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_item, parent, false);
        MyViewHolder myViewHolder = new MyViewHolder(chatView);
        return myViewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {
        Message message = messageList.get(position);
        
        if(message.getSentBy().equals(Message.SENT_BY_ME)){
            // 用户消息 - 显示在右侧
            holder.leftChatView.setVisibility(View.GONE);
            holder.rightChatView.setVisibility(View.VISIBLE);
            holder.rightChatTextView.setText(message.getMessage());
        } else {
            // Bot消息 - 显示在左侧
            holder.rightChatView.setVisibility(View.GONE);
            holder.leftChatView.setVisibility(View.VISIBLE);
            
            // 检查是否有结构化数据
            if (message.hasStructuredData()) {
                // 显示结构化消息
                holder.leftChatTextView.setVisibility(View.GONE);
                holder.reasoningContainer.setVisibility(View.VISIBLE);
                holder.actionContainer.setVisibility(View.VISIBLE);
                
                // 设置思维链内容
                holder.reasoningContent.setText(message.getReasoning());
                holder.reasoningContent.setTextIsSelectable(true); // 允许长按选中文字进行复制
                
                // 设置执行动作内容
                StringBuilder actionText = new StringBuilder();
                actionText.append(message.getAction());
                
                if (message.getParameters() != null && !message.getParameters().isEmpty()) {
                    actionText.append("\n").append(message.getParameters());
                }
                
                holder.actionContent.setText(actionText.toString());
                holder.actionContent.setTextIsSelectable(true); // 允许长按选中文字进行复制
                
                // 设置思维链折叠/展开功能
                setupReasoningToggle(holder);
                
            } else {
                // 显示简单文本消息
                holder.leftChatTextView.setVisibility(View.VISIBLE);
                holder.leftChatTextView.setText(message.getMessage());
                holder.leftChatTextView.setTextIsSelectable(true); // 允许长按选中文字进行复制
                holder.reasoningContainer.setVisibility(View.GONE);
                holder.actionContainer.setVisibility(View.GONE);
            }
        }
    }
    
    /**
     * 设置思维链的展开/折叠功能
     */
    private void setupReasoningToggle(MyViewHolder holder) {
        // 默认展开深度思考
        holder.reasoningContentWrapper.setVisibility(View.VISIBLE);
        holder.reasoningToggle.setText("▲");
        
        // 点击标题栏展开/折叠（直接切换，无动画）
        holder.reasoningHeader.setOnClickListener(v -> {
            if (holder.reasoningContentWrapper.getVisibility() == View.GONE) {
                // 展开
                holder.reasoningContentWrapper.setVisibility(View.VISIBLE);
                holder.reasoningToggle.setText("▲");
            } else {
                // 折叠
                holder.reasoningContentWrapper.setVisibility(View.GONE);
                holder.reasoningToggle.setText("▼");
            }
        });
    }

    @Override
    public int getItemCount() {
        return messageList.size();
    }

    public class MyViewHolder extends RecyclerView.ViewHolder{
        LinearLayout leftChatView, rightChatView;
        TextView leftChatTextView, rightChatTextView;
        
        // 结构化消息的视图组件
        LinearLayout reasoningContainer, actionContainer;
        LinearLayout reasoningHeader;
        RelativeLayout reasoningContentWrapper;
        TextView reasoningTitle, reasoningContent, reasoningToggle;
        TextView actionContent;

        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            leftChatView = itemView.findViewById(R.id.left_chat_view);
            rightChatView = itemView.findViewById(R.id.right_chat_view);
            leftChatTextView = itemView.findViewById(R.id.left_chat_text_view);
            rightChatTextView = itemView.findViewById(R.id.right_chat_text_view);
            
            // 初始化结构化消息组件
            reasoningContainer = itemView.findViewById(R.id.reasoning_container);
            actionContainer = itemView.findViewById(R.id.action_container);
            reasoningHeader = itemView.findViewById(R.id.reasoning_header);
            reasoningTitle = itemView.findViewById(R.id.reasoning_title);
            reasoningContentWrapper = itemView.findViewById(R.id.reasoning_content_wrapper);
            reasoningContent = itemView.findViewById(R.id.reasoning_content);
            reasoningToggle = itemView.findViewById(R.id.reasoning_toggle);
            actionContent = itemView.findViewById(R.id.action_content);
        }
    }
}
