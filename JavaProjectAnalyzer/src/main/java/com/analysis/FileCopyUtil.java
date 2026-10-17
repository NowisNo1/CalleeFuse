package com.analysis;

import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FileCopyUtil {

    public static void copyDirectory(String sourceDir, String targetDir) throws IOException{
        File source = new File(sourceDir);
        File target = new File(targetDir);

        if(!source.exists() || !source.isDirectory()){
            throw new IOException();
        }

        if(!target.exists()){
            boolean created = target.mkdirs();
            if(!created){
                throw new IOException();
            }
        }

        File[] files = source.listFiles();
        if(files == null){
            throw new IOException();
        }

        for(File file: files){
            File targetFile = new File(target, file.getName());
            if(file.isDirectory()){
                copyDirectory(file.getAbsolutePath(), targetFile.getAbsolutePath());
            }else{
                copyFile(file, targetFile);
            }
        }
    }
    private static void copyFile(File sourceFile, File targetFile) throws IOException {
        if(targetFile.exists()){
            System.out.println("copy");
        }

        try(FileChannel sourceChannel = new FileInputStream(sourceFile).getChannel();
            FileChannel targetChannel = new FileOutputStream(targetFile).getChannel()){

            targetChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
        }catch (IOException e){
            throw new IOException();
        }
    }
    public static void replaceLine(String filePath, int startLine, int endLine, String newContent) throws IOException{
        File sourceFile = new File(filePath);

        if(!sourceFile.exists() || !sourceFile.isFile()){
            throw new FileNotFoundException();
        }

        File tempFile = File.createTempFile("tmp_range_replace_", ".tmp", sourceFile.getParentFile());
        tempFile.deleteOnExit();

        List<String> newLines = new ArrayList<>();
        if(newContent.equals("")){
            for(int i = startLine; i <= endLine; i++){
                newLines.add("");
            }
        }else{
            newLines = Arrays.asList(newContent.split("\\r?\\n"));
        }
        try(BufferedReader reader = new BufferedReader(new FileReader(sourceFile));
            BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile))){
            String line;
            int idx = 0;
            int current = 1;
            while((line = reader.readLine()) != null){
                if(current >= startLine && current <= endLine){
                    writer.write(newLines.get(idx));
                    writer.newLine();
                    idx++;
                }else{
                    writer.write(line);
                    writer.newLine();
                }
                current++;
            }
            if(startLine > endLine || endLine > current - 1){
                throw new IllegalArgumentException();
            }
        }catch (IOException e){
            throw new IOException();
        }

        Files.move(
            tempFile.toPath(),
            Paths.get(filePath),
            StandardCopyOption.REPLACE_EXISTING,
            StandardCopyOption.ATOMIC_MOVE
        );
    }
    public static void deleteEmptyFolders(String rootDir) throws IOException{
        Path rootPath = Paths.get(rootDir);
        if(!Files.exists(rootPath)){
            return;
        }
        if(!Files.isDirectory(rootPath)){
            return;
        }

        Files.walkFileTree(rootPath, new SimpleFileVisitor<Path>(){
            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException{
                if(dir.equals(rootPath)){
                    return FileVisitResult.CONTINUE;
                }

                if(isDirectoryEmpty(dir)){
                    Files.delete(dir);
                }
                return FileVisitResult.CONTINUE;
            }
        });
    }

    private static boolean isDirectoryEmpty(Path dir) throws IOException {
        try(DirectoryStream<Path> stream = Files.newDirectoryStream(dir)){
            return !stream.iterator().hasNext();
        }
    }
}
