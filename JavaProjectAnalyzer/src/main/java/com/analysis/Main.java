package com.analysis;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.Range;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithName;
import com.github.javaparser.resolution.TypeSolver;
import com.github.javaparser.resolution.declarations.*;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.javaparsermodel.declarations.JavaParserMethodDeclaration;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import com.entity.Method;

import static com.analysis.FileCopyUtil.deleteEmptyFolders;

public class Main {
    final static boolean RANDOM_MASK = false;
    final static ArrayList<String> RANDOM_NAMES = new ArrayList<>();
    final static String jdkPath = "/path/to/jdk";
    final static String projectName = "zaproxy";
    final static boolean COPY_AND_CHANGE_ORIGIN_FILE = false;
    final static boolean USE_MASK_PROJECT = false;
    final static String ORIGIN_PATH = "/path/to/JavaProjects/";
    final static HashSet<String> REMOVE = new HashSet<>();
    final static String MASKED_PATH = "/path/to/JavaProjectAnalyzer/maskedProjects/";
    final static String projectPath = (USE_MASK_PROJECT && !COPY_AND_CHANGE_ORIGIN_FILE) ? MASKED_PATH + projectName : ORIGIN_PATH + projectName;
    final static String COPY_PATH = MASKED_PATH + projectName;
    private static final HashMap<String, String> MASKED = new HashMap<>();
    private static final Map<String, String> METHOD_SIGNATURE_MAP = new HashMap<>();
    private static final Map<String, MethodDeclaration> METHOD_MAP = new HashMap<>();
    private static final Set<String> VISITED_METHOD = new HashSet<>();
    private static final Set<String> VISITED = new HashSet<>();
    private static final List<Method> METHODS = new ArrayList<>();
    private static final List<String> HASH_CODES = new ArrayList<>();
    private static final Map<File, CompilationUnit> CU_CACHE = new HashMap<>();
    private static final List<String> sourceRoots = new ArrayList<>();
    private static final int MIN_CHILD = 1;
    private static final int MAX_DEPTH = 3;
    private static final JSONArray jsonArray = new JSONArray();
    private static final boolean MERGE = true;
    public static void main(String[] args) throws Exception{
        if(MERGE){
            merge();
        }else{
            readInvalid();
            File rootDir = new File(projectPath);
            addAllFiles(rootDir);
            if(COPY_AND_CHANGE_ORIGIN_FILE){
                FileCopyUtil.copyDirectory(projectPath, COPY_PATH);
            }
            findSourceRoots(rootDir, "src/main/java");
            for(String sourcePath: sourceRoots){
                File sourceDir = new File(sourcePath);
                if(!sourceDir.exists() || !sourceDir.isDirectory()){
                    System.err.println("source code directory is not exists :" + sourceDir.getAbsolutePath());
                }
                initSymbolSolver(sourceDir);
                scanAllMethodDefinitions(sourceDir);
                analyzeCrossFileMethodCalls(sourceDir);
                if(COPY_AND_CHANGE_ORIGIN_FILE){
                    changeOriginFile();
                }else{
                    printCallRelations();
                }

                clear();
            }
            if(!COPY_AND_CHANGE_ORIGIN_FILE){
                BufferedWriter writer;

                try{
                    String jsonStr = jsonArray.toJSONString(JSONWriter.Feature.PrettyFormat, JSONWriter.Feature.LargeObject);

                    writer = new BufferedWriter(new FileWriter(projectName + ".json"));
                    writer.write(jsonStr);
                    writer.flush();
                    writer.close();
                }catch (IOException ignored){

                }
            }else{
                try {
                    for(String filePath: REMOVE){
                        Files.deleteIfExists(Paths.get(filePath));
                    }
                    deleteEmptyFolders(COPY_PATH);
                }catch (Exception ignore){

                }
            }
        }
    }

    private static void changeOriginFile() {
        if(METHODS.isEmpty()){
            System.out.println("empty");
            return;
        }
        int num = 0;
        for(int i = 0; i < METHODS.size(); i++){

            Method caller =  METHODS.get(i);
            REMOVE.remove(caller.getFilePath().replace(ORIGIN_PATH, MASKED_PATH));
            if(caller.getMethodName().equals("main")){
                continue;
            }
            List<Method> callRelation = caller.getChildren();
            if(callRelation.size() >= MIN_CHILD){
                String rootMask = "ExtraId" + num;
                if(!MASKED.containsKey(caller.getHashCode())){
                    num += 1;
                }else{
                    rootMask = MASKED.get(caller.getHashCode());
                }
                num = parseChild(caller, num, rootMask, 1, MAX_DEPTH, "ExtraId");
            }else if(!MASKED.containsKey(caller.getHashCode())){
                try{
                    FileCopyUtil.replaceLine(caller.getFilePath().replace(ORIGIN_PATH, MASKED_PATH),
                            caller.getDeclarationRange().begin.line,
                            caller.getDeclarationRange().end.line,
                            "");

                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }

    }

    private static void clear(){
        METHOD_SIGNATURE_MAP.clear();
        METHOD_MAP.clear();
        CU_CACHE.clear();

        VISITED_METHOD.clear();

        METHODS.clear();
        HASH_CODES.clear();

        MASKED.clear();

        RANDOM_NAMES.clear();
    }
    private static void findSourceRoots(File dir, String sourcePath) {
        File srcMainJava = new File(dir, sourcePath);
        if (srcMainJava.exists() && srcMainJava.isDirectory()) {
            sourceRoots.add(srcMainJava.getAbsolutePath());
        }

        File[] subDirs = dir.listFiles(File::isDirectory);
        if (subDirs != null) {
            for (File subDir : subDirs) {
                findSourceRoots(subDir, sourcePath);
            }
        }
    }
    private static void readInvalid(){
        try (Stream<String> lines = Files.lines(Paths.get("invalidPaths"))) {
            lines.forEach(VISITED::add);
        } catch (IOException ignored) {

        }

    }


    private static void merge(){
        String filePath = "projectNames";

        try{
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String line;
            JSONArray mergeData = new JSONArray();
            while((line = br.readLine()) != null){
                String projectName = line.strip() + ".json";
                JSONArray projectData = JSONArray.parse(Files.readString(Paths.get(projectName)));
                mergeData.addAll(projectData);
            }

            br.close();

            try{
                String jsonStr = mergeData.toJSONString(JSONWriter.Feature.PrettyFormat, JSONWriter.Feature.LargeObject);
                BufferedWriter writer = new BufferedWriter(new FileWriter("mergeDataWithNoTestMethodMaskSubMethod.json"));
                writer.write(jsonStr);
                writer.flush();
                writer.close();
            }catch (IOException ignored){

            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
    public static String getSecureHash(String str){
        try{
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(str.getBytes(StandardCharsets.UTF_8));

            StringBuilder sb = new StringBuilder();
            for(byte b: hashBytes){
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        }catch (NoSuchAlgorithmException e){
            throw new RuntimeException("hash algorithm not exist");
        }
    }

    private static void initSymbolSolver(File sourceDir) throws IOException {
        TypeSolver jdkTypeSolver = new JarTypeSolver(jdkPath);
        CombinedTypeSolver projectTypeSolver = new CombinedTypeSolver();

        projectTypeSolver.add(new JavaParserTypeSolver(sourceDir));
        projectTypeSolver.add(jdkTypeSolver);
        projectTypeSolver.add(new ReflectionTypeSolver());

        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(projectTypeSolver);
        StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);
        StaticJavaParser.getParserConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_21);
    }
    private static String getSignature(String callerPackage, MethodDeclaration method){
        List<String> outerClassNames = new ArrayList<>();

        Node currentNode = method.getParentNode().orElse(null);
        while(!(currentNode instanceof ClassOrInterfaceDeclaration) && currentNode != null){
            currentNode = currentNode.getParentNode().orElse(null);
        }
        while(currentNode instanceof ClassOrInterfaceDeclaration){
            ClassOrInterfaceDeclaration outerClass = (ClassOrInterfaceDeclaration) currentNode;
            outerClassNames.add(outerClass.getNameAsString());
            currentNode = outerClass.getParentNode().orElse(null);
        }
        StringBuilder fullName = new StringBuilder();
        if(!callerPackage.isEmpty()){
            fullName.append(callerPackage).append(".");
        }
        for(int i = outerClassNames.size() - 1; i >= 0; i--){
            fullName.append(outerClassNames.get(i)).append(".");
        }
        fullName.deleteCharAt(fullName.length() - 1);
        String fullCallerClassName = fullName.toString();

        return buildFullMethodSignature(method, fullCallerClassName);
    }
    private static void addAllFiles(File dir){
        File[] files = dir.listFiles();
        if(files == null) return;

        for(File file: files){
            if(file.isDirectory()){
                addAllFiles(file);
            }
            else{
                REMOVE.add(file.getAbsolutePath().replace(ORIGIN_PATH, MASKED_PATH));
            }
        }
    }
    private static void scanAllMethodDefinitions(File dir){
        File[] files = dir.listFiles();
        if(files == null) return;

        for(File file: files){
            if(file.isDirectory()){
                scanAllMethodDefinitions(file);
            }
            else{
                if(file.getName().endsWith(".java")){
                    try{
                        CompilationUnit cu = parseFileToCU(file);

                        String callerPackage = cu.getPackageDeclaration().map(NodeWithName::getNameAsString).orElse("");

                        cu.findAll(MethodDeclaration.class).forEach(methodCall -> {
                            String methodSignature = getSignature(callerPackage, methodCall);
                            String hashCode = getSecureHash(methodSignature + methodCall);
                            METHOD_SIGNATURE_MAP.put(hashCode, file.getAbsolutePath());
                            METHOD_MAP.put(hashCode, methodCall);

                            if(RANDOM_MASK){
                                RANDOM_NAMES.add(methodCall.getNameAsString());
                            }
                        });
                    }catch (Exception ignored){
                    }
                }

            }
        }
    }
    private static String getOriginalCode(List<String> originalLines, Optional<Range> range){
        int startLineIndex = range.get().begin.line - 1;
        int endLineIndex = range.get().end.line - 1;

        StringBuilder originalCode = new StringBuilder();
        String startLineContent = originalLines.get(startLineIndex);
        originalCode.append(startLineContent);
        for(int i = startLineIndex + 1; i < endLineIndex; i++){
            originalCode.append("\n").append(originalLines.get(i));
        }
        if(startLineIndex != endLineIndex){
            String endLineContent = originalLines.get(endLineIndex);
            originalCode.append("\n").append(endLineContent);
        }

        return originalCode.toString();
    }
    private static void parseChild(MethodDeclaration method, String callerMethodSignature, Method methodInfo, int depth, int threshold){
        if(depth == threshold){
            return;
        }
        ResolvedMethodDeclaration resolvedCallerMethod = method.resolve();
        List<MethodCallExpr> methodCalls = method.findAll(MethodCallExpr.class);
        for (MethodCallExpr methodCall : methodCalls) {
            try {
                ResolvedMethodDeclaration calleeResolvedMethod = methodCall.resolve();
                if (!(methodCall.getNameAsString().equals(calleeResolvedMethod.getName()) && methodCall.getArguments().size() == calleeResolvedMethod.getNumberOfParams())) {
                    continue;
                }
                if (!calleeResolvedMethod.isAbstract() && calleeResolvedMethod.getNumberOfParams() == methodCall.getArguments().size() && calleeResolvedMethod.getName().equals(methodCall.getNameAsString())) {
                    boolean isRecursive = false;
                    JavaParserMethodDeclaration calleeMethodDecl = (JavaParserMethodDeclaration) calleeResolvedMethod;
                    String calleeMethodSignature = buildFullMethodSignature(calleeResolvedMethod);
                    if (method.getNameAsString().equals(methodCall.getNameAsString()) && methodCall.getArguments().size() == method.getParameters().size()) {
                        if (callerMethodSignature.equals(calleeMethodSignature) &&
                                method.getName().toString().equals(calleeResolvedMethod.getName()) &&
                                resolvedCallerMethod.getTypeParameters().stream().map(ResolvedTypeParameterDeclaration::getName).collect(Collectors.joining(","))
                                        .equals(calleeResolvedMethod.getTypeParameters().stream().map(ResolvedTypeParameterDeclaration::getName).collect(Collectors.joining(",")))) {
                            isRecursive = true;
                        }
                    }
                    String hashCodeCallee = getSecureHash(calleeMethodSignature + calleeMethodDecl.getWrappedNode());
                    if (METHOD_SIGNATURE_MAP.containsKey(hashCodeCallee)) {
                        File calleeFile = new File(METHOD_SIGNATURE_MAP.get(hashCodeCallee));
                        List<String> calledOriginalLines = new ArrayList<>();
                        try (BufferedReader br = new BufferedReader(new FileReader(calleeFile))) {
                            String line;
                            while ((line = br.readLine()) != null) {
                                calledOriginalLines.add(line);
                            }
                        }
                        String calleeMethodBody = getOriginalCode(calledOriginalLines, calleeMethodDecl.getWrappedNode().getRange());
                        Method calleeInfo = new Method(hashCodeCallee,
                                                        calleeFile.getAbsolutePath(),
                                                        calleeMethodSignature,
                                                        calleeMethodBody,
                                                        calleeMethodDecl.getWrappedNode().getRange().get(),
                                                        methodCall.getRange().get(),
                                                        methodCall.getNameAsString(),
                                                        removeComments(calleeMethodDecl.getWrappedNode().toString()),
                                                        isRecursive,
                                                        new ArrayList<>());

                        parseChild(METHOD_MAP.get(hashCodeCallee), calleeMethodSignature, calleeInfo, depth + 1, threshold);
                        methodInfo.getChildren().add(calleeInfo);
                    }
                }
            } catch (Exception ignored) {
            }
        }
    }
    private static void parseJavaFile(File file){
        try{
            if(VISITED.contains(file.getAbsolutePath())){
                return;
            }
            CompilationUnit cu = parseFileToCU(file);
            List<String> originalLines = new ArrayList<>();
            try(BufferedReader br = new BufferedReader(new FileReader(file))){
                String line;
                while((line = br.readLine()) != null){
                    originalLines.add(line);
                }
            }
            VISITED.add(file.getAbsolutePath());
            String callerPackage = cu.getPackageDeclaration().map(NodeWithName::getNameAsString).orElse("");
            List<MethodDeclaration> callerMethods = cu.findAll(MethodDeclaration.class);
            System.out.println(file.getAbsolutePath());
            for (MethodDeclaration method : callerMethods) {
                try {
                    if (!method.isAbstract()) {
                        String callerMethodSignature = getSignature(callerPackage, method);
                        String hashCode = getSecureHash(callerMethodSignature + method);
                        if (VISITED_METHOD.contains(hashCode)) continue;
                        VISITED_METHOD.add(hashCode);
                        String callerFilePath = METHOD_SIGNATURE_MAP.get(hashCode);
                        if (callerFilePath == null) continue;

                        String callerMethodBody = getOriginalCode(originalLines, method.getRange());
                        HASH_CODES.add(hashCode);
                        Method callerInfo = new Method(
                                hashCode,
                                callerFilePath,
                                callerMethodSignature,
                                callerMethodBody,
                                method.getRange().get(),
                                null,
                                method.getNameAsString(),
                                removeComments(method.toString()),
                                false,
                                new ArrayList<>());
                        parseChild(METHOD_MAP.get(hashCode), callerMethodSignature, callerInfo, 0, MAX_DEPTH - 1);
                        METHODS.add(callerInfo);
                    }
                } catch (Exception ignored) {
                }
            }
        }catch (Exception ignored){
        }
    }
    private static JSONObject parseChild(Method methodInfo, JSONArray children, int num, int depth, int threshold, String base){
        JSONObject ret = new JSONObject();
        if(depth == threshold){
            ret.put("num", num);
            ret.put("children", children);
            return ret;
        }
        Range callerRange = methodInfo.getDeclarationRange();
        List<Range> calleeRanges = new ArrayList<>();
        List<String> calleeNames = new ArrayList<>();
        List<String> masks = new ArrayList<>();
        List<String> randomMasks = new ArrayList<>();
        int rootNum = num - 1;

        for (Method callee : methodInfo.getChildren()) {

            String mask;
            if(callee.isRecursive()){
                if(depth != 0){
                    mask = base + rootNum + ">";
                }
                else{
                    mask = "<mask>";
                }
            }
            else{
                mask = base + num + ">";
            }

            if(depth == 0 || callee.isRecursive()){
                masks.add(mask);
                calleeRanges.add(callee.getCallExprRange());
                calleeNames.add(callee.getMethodName());
            }
            if(callee.isRecursive()){
                continue;
            }

            JSONObject resp = parseChild(callee, new JSONArray(), num + 1, depth + 1, threshold, base);
            num = (int) resp.get("num");
            if(!resp.get("method_body").equals("<error>")){
                JSONObject child = new JSONObject();
                child.put("method_full_name", callee.getSignature());
                String methodBody = (String) resp.get("method_body");
                if(RANDOM_MASK){
                    Random random = new Random();
                    if(RANDOM_NAMES.size() > 0){
                        int randomIndex = random.nextInt(RANDOM_NAMES.size());
                        methodBody = methodBody.replace(mask, RANDOM_NAMES.get(randomIndex));
                        randomMasks.add(RANDOM_NAMES.get(randomIndex));
                    }
                }

                child.put("method_body", methodBody);
                child.put("children", resp.get("children"));
                children.add(child);
            }
            else{
                masks.remove(masks.size() - 1);
                calleeRanges.remove(calleeRanges.size() - 1);
                calleeNames.remove(calleeNames.size() - 1);
            }
        }
        List<LinePosition> linePositions = parseLinePositions(methodInfo.getMethodBody());
        String maskedMethod = maskSubMethod(calleeNames, callerRange, calleeRanges, methodInfo.getMethodBody(), linePositions, masks);
        if(!maskedMethod.equals("<error>")){
            if(depth == 0){
                maskedMethod = maskMethodName(methodInfo.getMethodName(), maskedMethod, "<mask>");
                if(RANDOM_MASK){

                    ArrayList<String> maskExcludeRecursive = new ArrayList<>();
                    for (String mask : masks) {
                        if (!mask.equals("<mask>")) {
                            maskExcludeRecursive.add(mask);
                        }
                    }
                    for (int i = 0; i < randomMasks.size(); i++) {
                        if(maskedMethod.contains(maskExcludeRecursive.get(i))){
                            maskedMethod = maskedMethod.replaceAll(maskExcludeRecursive.get(i), Matcher.quoteReplacement(randomMasks.get(i)));
                        }
                    }
                }
            }
            else{
                if(depth + 1 <= threshold){
                    String rootMask = base + rootNum + ">";

                    maskedMethod = maskMethodName(methodInfo.getMethodName(), maskedMethod, rootMask);
                }

            }
            maskedMethod = removeComments(eliminateIndentation(maskedMethod, methodInfo));
        }

        ret.put("num", num);
        ret.put("method_body", maskedMethod);
        ret.put("children", children);

        return ret;

    }
    private static int parseChild(Method methodInfo, int num, String rootMask, int depth, int threshold, String base){
        if(depth == threshold){
            return num;
        }
        Range callerRange = methodInfo.getDeclarationRange();
        List<Range> calleeRanges = new ArrayList<>();
        List<String> calleeNames = new ArrayList<>();
        List<String> masks = new ArrayList<>();

        for (Method callee : methodInfo.getChildren()) {
            String mask = base + num;
            if(MASKED.containsKey(callee.getHashCode())){
                mask = MASKED.get(callee.getHashCode());
            }else{
                num++;
                MASKED.put(callee.getHashCode(), mask);
            }
            String calleeBody = maskMethodName(callee.getMethodName(), callee.getMethodBody(), mask);

            if(calleeBody.equals("<error>")){
                continue;
            }
            try{
                FileCopyUtil.replaceLine(callee.getFilePath().replace(ORIGIN_PATH, MASKED_PATH),
                        callee.getDeclarationRange().begin.line,
                        callee.getDeclarationRange().end.line,
                        calleeBody);

            }catch (Exception e){
                e.printStackTrace();
            }

            masks.add(mask);

            calleeRanges.add(callee.getCallExprRange());
            calleeNames.add(callee.getMethodName());
            num = parseChild(callee, num, mask,depth + 1, threshold, base);

        }
        List<LinePosition> linePositions = parseLinePositions(methodInfo.getMethodBody());
        String maskedMethod = maskSubMethod(calleeNames, callerRange, calleeRanges, methodInfo.getMethodBody(), linePositions, masks);

        if(MASKED.containsKey(methodInfo.getHashCode())){
            maskedMethod = maskMethodName(methodInfo.getMethodName(), maskedMethod, MASKED.get(methodInfo.getHashCode()));
        }else{
            maskedMethod = maskMethodName(methodInfo.getMethodName(), maskedMethod, rootMask);
            MASKED.put(methodInfo.getHashCode(), rootMask);
        }

        try{
            FileCopyUtil.replaceLine(methodInfo.getFilePath().replace(ORIGIN_PATH, MASKED_PATH),
                    methodInfo.getDeclarationRange().begin.line,
                    methodInfo.getDeclarationRange().end.line,
                    maskedMethod);

        }catch (Exception e){
            e.printStackTrace();
        }
        return num;
    }
    private static void analyzeCrossFileMethodCalls(File dir){
        File[] files = dir.listFiles();
        if(files == null) return;

        for(File file: files){
            if(file.isDirectory()){
                analyzeCrossFileMethodCalls(file);
            }
            else if(file.getName().endsWith(".java")){
                parseJavaFile(file);
            }
        }
    }
    private static final ThreadLocal<StringBuilder> COMMENT_BUILDER = ThreadLocal.withInitial(StringBuilder::new);
    public static String removeComments(String methodBody) {
        if (methodBody == null || methodBody.isEmpty()) {
            return methodBody;
        }

        StringBuilder result = COMMENT_BUILDER.get();
        result.setLength(0);
        int length = methodBody.length();
        int i = 0;
        boolean inString = false;
        boolean inBlockComment = false;

        while (i < length) {
            char current = methodBody.charAt(i);
            if (inBlockComment) {
                if (current == '*' && i + 1 < length && methodBody.charAt(i + 1) == '/') {
                    inBlockComment = false;
                    i += 2;
                } else {
                    i++;
                }
                continue;
            }

            if (current == '"') {
                if (i == 0 || methodBody.charAt(i - 1) != '\\') {
                    inString = !inString;
                }
                result.append(current);
                i++;
                continue;
            }

            if (!inString) {
                if (current == '/' && i + 1 < length && methodBody.charAt(i + 1) == '/') {
                    while (i < length && methodBody.charAt(i) != '\n') {
                        i++;
                    }
                    continue;
                }
                else if (current == '/' && i + 1 < length && methodBody.charAt(i + 1) == '*') {
                    inBlockComment = true;
                    i += 2;
                    continue;
                }
            }
            result.append(current);
            i++;
        }
        String cleaned = result.toString();
        COMMENT_BUILDER.remove();
        return cleaned;
    }
    private static String buildFullMethodSignature(ResolvedMethodDeclaration method){
        String fullClassName = method.declaringType().getQualifiedName();

        String methodName = method.getName();

        List<String> paramTypes = new ArrayList<>();
        for(int i = 0; i < method.getNumberOfParams(); i++){
            ResolvedParameterDeclaration param = method.getParam(i);
            paramTypes.add(param.getType().describe());
        }
        String paramTypeStr = String.join(",", paramTypes);

        return String.format("%s.%s(%s)", fullClassName, methodName, paramTypeStr);
    }
    private static String buildFullMethodSignature(MethodDeclaration method, String fullClassName){
        String methodName = method.getName().toString();

        List<String> paramTypes = new ArrayList<>();
        int length = method.getParameters().size();
        for(int i = 0; i < length; i++){
            Parameter param = method.getParameter(i);
            paramTypes.add(param.getType().toString());
        }
        String paramTypeStr = String.join(",", paramTypes);

        return String.format("%s.%s(%s)", fullClassName, methodName, paramTypeStr);
    }
    private static CompilationUnit parseFileToCU(File file) throws IOException{
        if(CU_CACHE.containsKey(file)){
            return CU_CACHE.get(file);
        }

        CompilationUnit cu = StaticJavaParser.parse(new FileInputStream(file));
        CU_CACHE.put(file, cu);
        return cu;
    }
    private static String maskMethodName(String methodName, String method, String mask){
        String re = " " + methodName + "(";
        String escapedMatchStr = Pattern.quote(re);

        String matchStr;
        if(method.charAt(0) == '@'){
            String prefix = method.substring(0, method.indexOf("\n") + 1);
            String suffix = method.substring(method.indexOf("\n") + 1);

            String newSuffix = suffix.replaceFirst(escapedMatchStr, " " + mask + "(");

            matchStr = prefix + newSuffix;

        }else {
            matchStr = method.replaceFirst(escapedMatchStr, " " + mask + "(");
        }
        return matchStr;
    }
    private static String eliminateIndentation(String methodBody, Method method){
        String[] subString = methodBody.split("\n");

        for(int j = 0; j < subString.length; j++){
            if(subString[j].length() < method.getDeclarationRange().begin.column){
                continue;
            }
            subString[j] = subString[j].substring(method.getDeclarationRange().begin.column - 1);
        }
        return String.join("\n", subString);
    }
    private static void printCallRelations(){
        if(METHODS.isEmpty()){
            System.out.println("empty");
            return;
        }
        for(int i = 0; i < METHODS.size(); i++) {
            Method caller = METHODS.get(i);

            JSONObject root = new JSONObject();
            root.put("method_full_name", caller.getSignature());

            if (caller.getMethodName().equals("main")) {
                continue;
            }
            List<Method> callRelation = caller.getChildren();
            if (callRelation.size() < MIN_CHILD) {
                continue;
            }

            JSONObject resp = parseChild(caller, new JSONArray(), 0, 0, MAX_DEPTH, "<extra_id_");
            if(!resp.get("method_body").equals("<error>")){
                root.put("hashCode", HASH_CODES.get(i));
                root.put("method_body", resp.get("method_body"));
                root.put("children", resp.get("children"));
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("root", root);
                jsonArray.add(jsonObject);
            }

        }
    }
    private static int[] rangeToStringIndex(List<LinePosition> linePositions, String text, int startLine, int startCol, int endLine, int endCol){
        int startIndex = calculateIndex(text, linePositions, startLine, startCol, "start");

        int endIndex = calculateIndex(text, linePositions, endLine, endCol, "end");

        return new int[]{startIndex, endIndex};
    }
    private static class LinePosition{
        int start;
        int end;
        LinePosition(int start, int end){
            this.start = start;
            this.end = end;
        }
    }
    private static List<LinePosition> parseLinePositions(String text){
        List<LinePosition> linePositions = new ArrayList<>();
        int lineStart = 0;
        int textLength = text.length();

        for(int i = 0; i < textLength; i++){
            if(text.charAt(i) == '\n'){
                linePositions.add(new LinePosition(lineStart, i + 1));
                lineStart = i + 1;
            }
        }

        if(lineStart < textLength){
            linePositions.add(new LinePosition(lineStart, textLength));
        }

        return linePositions;
    }
    private static int calculateIndex(String text, List<LinePosition> linePositions, int lineNum, int colNum, String type){
        if(lineNum < 1 || lineNum > linePositions.size()){
            throw new IllegalArgumentException(
                    String.format("%s line %d index out of range, text line %d", type, lineNum, linePositions.size())
            );
        }

        int lineIndex = lineNum - 1;
        LinePosition linePos = linePositions.get(lineIndex);
        int lineStart = linePos.start;
        int lineEnd = linePos.end;

        int lineContentLength = lineEnd - lineStart;
        char lastChar = lineEnd <= lineStart ? ' ' : linePos.end < lineStart + lineContentLength ? ' ' : text.charAt(lineEnd - 1);
        if(lastChar == '\n'){
            lineContentLength -= 1;
        }

        if(colNum < 1 || colNum > lineContentLength + 1){
            throw new IllegalArgumentException("col index out of range");
        }

        return lineStart + (colNum - 1);
    }
    private static String maskSubMethod(List<String> names, Range callerRange, List<Range> calleeRanges, String method, List<LinePosition> linePositions, List<String> masks){
        StringBuilder sb = new StringBuilder();
        int startIndex = 0;

        Map<Integer, String> calleeMask = new HashMap<>();
        Map<Integer, String> nameMask = new HashMap<>();
        Map<Integer, String> maskString = new HashMap<>();

        List<Integer> keys = new ArrayList<>();
        try{
            for (int i = 0; i < names.size(); i++) {
                String name = names.get(i);
                String mask = masks.get(i);
                Range calleeRange = calleeRanges.get(i);
                int startLine = calleeRange.begin.line - callerRange.begin.line + 1;
                int startCol = calleeRange.begin.column;
                int endLine = calleeRange.end.line - callerRange.begin.line + 1;
                int endCol = calleeRange.end.column;

                int[] res = rangeToStringIndex(linePositions, method, startLine, startCol, endLine, endCol);
                String ret = replaceIndependentMethodName(method, res[0], res[1], name, mask);
                int idx = ret.lastIndexOf(mask);
                if(idx != -1){
                    calleeMask.put(idx, ret);
                    nameMask.put(idx, name);
                    maskString.put(idx, mask);
                    keys.add(idx);
                }
            }
            if(calleeMask.isEmpty()){
                return method;
            }else {
                keys.sort(Comparator.naturalOrder());

                for (Integer key : keys) {
                    try {
                        int idx = key;
                        sb.append(calleeMask.get(idx), startIndex, idx);
                        sb.append(maskString.get(idx));
                        startIndex = idx + nameMask.get(idx).length();
                    } catch (IndexOutOfBoundsException e) {
                        return "<error>";
                    }
                }
                if(startIndex < method.length() - 1){
                    sb.append(method, startIndex, method.length());
                }
            }
            return sb.toString();
        }catch (Exception e){
            return "<error>";
        }

    }
    private static String replaceIndependentMethodName(String original, int start, int end, String targetMethod, String mask){
        if (original == null || targetMethod == null){
            throw new IllegalArgumentException("the value of String can not be null");
        }
        if (targetMethod.isEmpty()){
            throw new IllegalArgumentException("targetMethod can not be empty");
        }
        int methodLen = targetMethod.length();
        int originalLen = original.length();
        if (start < 0 || end > originalLen || start > end){
            throw new IllegalArgumentException(
                    String.format("invalid range of index start=%d, end=%d, originalLength=%d",
                            start, end, originalLen)
            );
        }
        int maxSearchIndex = Math.min(end - methodLen, originalLen - methodLen - 1) - 1;
        if(maxSearchIndex < start){
            return original;
        }

        StringBuilder sb = new StringBuilder(original);
        for(int i = maxSearchIndex; i >= start; i--){
            boolean isMethodMatch = true;
            for(int j = 0; j < methodLen; j++){
                if(sb.charAt(i + j) != targetMethod.charAt(j)){
                    isMethodMatch = false;
                    break;
                }
            }
            if(!isMethodMatch){
                continue;
            }

            int methodEndIndex = i + methodLen;
            if(methodEndIndex >= sb.length() || sb.charAt(methodEndIndex) != '('){

                continue;
            }
            boolean isIndependentMethod = true;
            if(i > 0){
                char prevChar = sb.charAt(i - 1);
                if(Character.isLetterOrDigit(prevChar) || prevChar == '_' || prevChar == '$'){
                    isIndependentMethod = false;
                }
            }

            if(isIndependentMethod){
                sb.replace(i, methodEndIndex, mask);
            }
        }
        return sb.toString();
    }
}
