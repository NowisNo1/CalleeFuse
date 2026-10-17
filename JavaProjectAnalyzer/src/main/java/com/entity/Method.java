package com.entity;

import com.github.javaparser.Range;

import java.util.ArrayList;

public class Method{
    String filePath;
    String hashCode;
    String methodName;
    String signature;
    String parsedBody;
    String methodBody;
    Range declarationRange;
    Range callExprRange;
    ArrayList<Method> children;
    boolean isRecursive;

    public Method(String hashCode, String filePath, String signature, String methodBody, Range declarationRange, Range callExprRange, String methodName, String parsedBody, boolean isRecursive, ArrayList<Method> children){
        this.filePath = filePath;
        this.hashCode = hashCode;
        this.signature = signature;
        this.methodBody = methodBody;
        this.declarationRange = declarationRange;
        this.callExprRange = callExprRange;
        this.methodName = methodName;
        this.parsedBody = parsedBody;
        this.isRecursive = isRecursive;
        this.children = children;
    }
    public String getHashCode(){
        return hashCode;
    }
    public ArrayList<Method> getChildren(){
        return children;
    }
    public String getFilePath(){
        return filePath;
    }
    public String getSignature(){
        return signature;
    }
    public String getMethodBody(){
        return methodBody;
    }
    public Range getDeclarationRange() { return declarationRange; }
    public Range getCallExprRange() { return callExprRange; }
    public String getMethodName(){
        return methodName;
    }

    public boolean isRecursive(){
        return isRecursive;
    }

}
