from aurelius.benchmark.config import BenchmarkPipelineConfig, EvalConfig, FinetuneConfig


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkPipelineConfig()
        assert config.batch_target_size == 50
        assert config.batch_min_size == 30
        assert config.influence_min_batch_size == 30

    def test_finetune_defaults(self):
        config = FinetuneConfig()
        assert config.lora_rank == 16
        assert config.num_epochs == 3
        assert "q_proj" in config.lora_target_modules

    def test_eval_defaults(self):
        config = EvalConfig()
        assert config.num_scenarios == 500
        assert config.judge_model == "deepseek-chat"
