package com.example.backend.chainmaker;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

/**
 * Initializes the ChainMaker client at startup.
 */
@Slf4j
@Component
@Order(0)
public class ChainmakerInitRunner implements ApplicationRunner {

    @Override
    public void run(ApplicationArguments args) {
        if (InitClient.chainClient != null) {
            log.info("ChainMaker client already initialized.");
            return;
        }
        try {
            InitClient.initChainClient();
            log.info("ChainMaker client initialized successfully.");
        } catch (Exception e) {
            log.error("Failed to initialize ChainMaker client", e);
        }
    }
}

