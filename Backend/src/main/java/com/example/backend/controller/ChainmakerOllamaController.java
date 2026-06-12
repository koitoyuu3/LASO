package com.example.backend.controller;

import com.example.backend.result.Result;
import com.example.backend.service.ChainmakerOllamaService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import java.math.BigInteger;
import java.util.Map;

/**
 * ChainMaker + Ollama aggregation API.
 */
@Slf4j
@RestController
@RequestMapping("/api/chainmaker/ollama")
@Tag(name = "ChainMaker-Ollama", description = "Stores the prompt via the OracleAggregator contract and dispatches a Python agent by requestId to drive up to 50 node LLMs")
public class ChainmakerOllamaController {

    @Resource
    private ChainmakerOllamaService chainmakerOllamaService;

    @PostMapping("/prompt")
    @Operation(summary = "Only write the prompt on-chain and return the requestId")
    public Result<Map<String, Object>> storePrompt(@RequestParam String prompt) {
        try {
            return Result.success(chainmakerOllamaService.storePrompt(prompt));
        } catch (Exception e) {
            log.error("store chainmaker ollama prompt failed", e);
            return Result.error("On-chain write failed: " + e.getMessage());
        }
    }

    @PostMapping("/infer-by-request")
    @Operation(summary = "Read the prompt from the chain by requestId and dispatch Python agent node tasks by count")
    public Result<Map<String, Object>> inferByRequestId(
            @RequestParam Long requestId,
            @RequestParam(required = false) Integer count
    ) {
        try {
            return Result.success(chainmakerOllamaService.inferByRequestId(BigInteger.valueOf(requestId), count));
        } catch (Exception e) {
            log.error("chainmaker ollama infer-by-request failed, requestId={}, count={}", requestId, count, e);
            return Result.error("Inference failed: " + e.getMessage());
        }
    }

    @PostMapping("/infer")
    @Operation(summary = "Compatibility entry: write the prompt on-chain first, then dispatch Python agent node tasks by count")
    public Result<Map<String, Object>> infer(
            @RequestParam String prompt,
            @RequestParam(required = false) Integer count
    ) {
        try {
            return Result.success(chainmakerOllamaService.infer(prompt, count));
        } catch (Exception e) {
            log.error("chainmaker ollama infer failed", e);
            return Result.error("Inference failed: " + e.getMessage());
        }
    }

    @GetMapping("/result")
    @Operation(summary = "Query the prompt info stored on-chain by requestId")
    public Result<Map<String, Object>> queryResult(@RequestParam Long requestId) {
        try {
            return Result.success(chainmakerOllamaService.queryResult(BigInteger.valueOf(requestId)));
        } catch (Exception e) {
            log.error("query chainmaker ollama result failed, requestId={}", requestId, e);
            return Result.error("Query failed: " + e.getMessage());
        }
    }
}
