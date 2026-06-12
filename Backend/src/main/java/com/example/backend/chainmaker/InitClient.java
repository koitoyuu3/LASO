package com.example.backend.chainmaker;

import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.chainmaker.sdk.ChainClient;
import org.chainmaker.sdk.ChainManager;
import org.chainmaker.sdk.config.ChainClientConfig;
import org.chainmaker.sdk.config.NodeConfig;
import org.chainmaker.sdk.config.SdkConfig;
import org.chainmaker.sdk.utils.FileUtils;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.representer.Representer;

public class InitClient {

    static String SDK_CONFIG = "sdk_config.yml";
    private static final String RESOURCE_PREFIX = "src/main/resources/";

    public static ChainClient chainClient;
    static ChainManager chainManager;

    public static synchronized void initChainClient() throws Exception {
        if (chainClient != null) {
            return;
        }

        Representer representer = new Representer(new DumperOptions());
        representer.getPropertyUtils().setSkipMissingProperties(true);
        Yaml yaml = new Yaml(representer);
        InputStream in = InitClient.class.getClassLoader().getResourceAsStream(SDK_CONFIG);
        if (in == null) {
            throw new IllegalArgumentException("resource " + SDK_CONFIG + " not found.");
        }

        SdkConfig sdkConfig = yaml.loadAs(in, SdkConfig.class);
        in.close();

        normalizeChainClientConfigPaths(sdkConfig);

        for (NodeConfig nodeConfig : sdkConfig.getChainClient().getNodes()) {
            List<byte[]> tlsCaCertList = new ArrayList<>();
            if (nodeConfig.getTrustRootPaths() != null) {
                List<String> resolvedTrustRootPaths = new ArrayList<>();
                for (String rootPath : nodeConfig.getTrustRootPaths()) {
                    String resolvedRootPath = resolveExistingPath(rootPath, true);
                    resolvedTrustRootPaths.add(resolvedRootPath);
                    List<String> filePathList = FileUtils.getFilesByPath(resolvedRootPath);
                    for (String filePath : filePathList) {
                        tlsCaCertList.add(FileUtils.getFileBytes(filePath));
                    }
                }
                nodeConfig.setTrustRootPaths(resolvedTrustRootPaths.toArray(new String[0]));
            }
            byte[][] tlsCaCerts = new byte[tlsCaCertList.size()][];
            tlsCaCertList.toArray(tlsCaCerts);
            nodeConfig.setTrustRootBytes(tlsCaCerts);
        }

        chainManager = ChainManager.getInstance();
        chainClient = chainManager.getChainClient(sdkConfig.getChainClient().getChainId());

        if (chainClient == null) {
            chainClient = chainManager.createChainClient(sdkConfig);
        }

        System.out.println("init client success.");
    }

    private static void normalizeChainClientConfigPaths(SdkConfig sdkConfig) {
        ChainClientConfig chainClientConfig = sdkConfig.getChainClient();
        if (chainClientConfig == null) {
            return;
        }

        chainClientConfig.setUserKeyFilePath(resolveExistingPath(chainClientConfig.getUserKeyFilePath(), false));
        chainClientConfig.setUserCrtFilePath(resolveExistingPath(chainClientConfig.getUserCrtFilePath(), false));
        chainClientConfig.setUserSignKeyFilePath(resolveExistingPath(chainClientConfig.getUserSignKeyFilePath(), false));
        chainClientConfig.setUserSignCrtFilePath(resolveExistingPath(chainClientConfig.getUserSignCrtFilePath(), false));
    }

    private static String resolveExistingPath(String originalPath, boolean directoryExpected) {
        if (originalPath == null || originalPath.trim().isEmpty()) {
            return originalPath;
        }

        String trimmedPath = originalPath.trim();
        for (Path candidate : buildCandidates(trimmedPath)) {
            if (!Files.exists(candidate)) {
                continue;
            }
            if (directoryExpected && Files.isDirectory(candidate)) {
                return candidate.toAbsolutePath().normalize().toString();
            }
            if (!directoryExpected && Files.isRegularFile(candidate)) {
                return candidate.toAbsolutePath().normalize().toString();
            }
        }

        URL resourceUrl = resolveResourceUrl(trimmedPath);
        if (resourceUrl != null && "file".equalsIgnoreCase(resourceUrl.getProtocol())) {
            try {
                Path resourcePath = Paths.get(resourceUrl.toURI());
                if (Files.exists(resourcePath)) {
                    return resourcePath.toAbsolutePath().normalize().toString();
                }
            } catch (Exception ignored) {
                return trimmedPath;
            }
        }

        return trimmedPath;
    }

    private static List<Path> buildCandidates(String pathString) {
        List<Path> candidates = new ArrayList<>();
        Path rawPath = Paths.get(pathString);
        if (rawPath.isAbsolute()) {
            candidates.add(rawPath);
            return candidates;
        }

        Path currentDir = Paths.get("").toAbsolutePath().normalize();
        candidates.add(currentDir.resolve(pathString).normalize());
        candidates.add(currentDir.resolve("Backend").resolve(pathString).normalize());

        if (pathString.startsWith(RESOURCE_PREFIX)) {
            String resourceRelativePath = pathString.substring(RESOURCE_PREFIX.length());
            candidates.add(currentDir.resolve("Backend").resolve(RESOURCE_PREFIX).resolve(resourceRelativePath).normalize());
            candidates.add(currentDir.resolve(RESOURCE_PREFIX).resolve(resourceRelativePath).normalize());
        }

        return candidates;
    }

    private static URL resolveResourceUrl(String pathString) {
        ClassLoader classLoader = InitClient.class.getClassLoader();
        URL resourceUrl = classLoader.getResource(pathString);
        if (resourceUrl != null) {
            return resourceUrl;
        }
        if (pathString.startsWith(RESOURCE_PREFIX)) {
            String resourceRelativePath = pathString.substring(RESOURCE_PREFIX.length());
            return classLoader.getResource(resourceRelativePath);
        }
        return null;
    }
}
