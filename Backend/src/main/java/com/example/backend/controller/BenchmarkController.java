package com.example.backend.controller;

import com.example.backend.result.Result;
import com.example.backend.service.OllamaFixedResultBenchmarkService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import javax.annotation.Resource;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/benchmark")
@Tag(name = "Benchmark", description = "End-to-end latency benchmark with fixed AI output")
public class BenchmarkController {

    @Resource
    private OllamaFixedResultBenchmarkService ollamaFixedResultBenchmarkService;

    @PostMapping("/ollama-fixed-result")
    @Operation(summary = "Skip AI inference and use fixed agent output to measure prompt write, callback write, and result query latency")
    public Result<Map<String, Object>> runOllamaFixedResult(
            @RequestParam String chain,
            @RequestParam String resultPath,
            @RequestParam(required = false) String proofBundlePath,
            @RequestParam(required = false) String prompt,
            @RequestParam(required = false) Integer payloadPreviewLength
    ) {
        try {
            return Result.success(
                    ollamaFixedResultBenchmarkService.run(
                            chain,
                            prompt,
                            resultPath,
                            proofBundlePath,
                            payloadPreviewLength
                    )
            );
        } catch (Exception e) {
            log.error(
                    "fixed-result benchmark failed, chain={}, resultPath={}, proofBundlePath={}, prompt={}",
                    chain,
                    resultPath,
                    proofBundlePath,
                    prompt,
                    e
            );
            return Result.error("benchmark failed: " + e.getMessage());
        }
    }
}
