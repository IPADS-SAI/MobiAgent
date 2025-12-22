package com.mobi.agent;

public class Message {
    public static String SENT_BY_ME = "me";
    public static String SENT_BY_BOT = "bot";

    String message;
    String sentBy;
    
    // 新增字段：用于结构化展示
    String reasoning;      // AI思维链
    String action;         // 执行动作
    String parameters;     // 动作参数

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getSentBy() {
        return sentBy;
    }

    public void setSentBy(String sentBy) {
        this.sentBy = sentBy;
    }
    
    public String getReasoning() {
        return reasoning;
    }
    
    public void setReasoning(String reasoning) {
        this.reasoning = reasoning;
    }
    
    public String getAction() {
        return action;
    }
    
    public void setAction(String action) {
        this.action = action;
    }
    
    public String getParameters() {
        return parameters;
    }
    
    public void setParameters(String parameters) {
        this.parameters = parameters;
    }
    
    // 检查是否有结构化数据
    public boolean hasStructuredData() {
        return reasoning != null && !reasoning.isEmpty();
    }

    // 原始构造函数
    public Message(String message, String sentBy) {
        this.message = message;
        this.sentBy = sentBy;
    }
    
    // 新增构造函数：支持结构化数据
    public Message(String message, String sentBy, String reasoning, String action, String parameters) {
        this.message = message;
        this.sentBy = sentBy;
        this.reasoning = reasoning;
        this.action = action;
        this.parameters = parameters;
    }
}
